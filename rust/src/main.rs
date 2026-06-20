//! Gaia robot console application.
//!
//! This is the single orchestrating entry point for the program. Read it
//! top-to-bottom to follow the entire flow, much like a C# console app's
//! `Main` method. The detailed data of each "class" lives in its own module
//! (one type per file); `main` only wires those pieces together.
//!
//! # The eleven-block flow
//!
//! The architecture diagram (`Gaia Physical Architecture.drawio.png`) models the
//! program as a loop of eleven blocks. Conceptually a turn is two LLM passes
//! with a deliberate division of labour:
//!
//! - **LLM Call 1 is the *pull* pass.** It reads Gaia's running context plus the
//!   user's new sentence, decides *what information to gather*, and emits only
//!   **read / non-side-effecting** actions (web search and semantic/logical
//!   index queries). Those actions are run by their applets and the results are
//!   collected into an in-memory store called the **Response Data Context**.
//! - **LLM Call 2 is the *push* pass.** It reads the compressed carry-over from
//!   Call 1 plus the Response Data Context, *answers the user*, and emits
//!   **side-effecting** actions (send WhatsApp/Push, actuate, emote, and
//!   write-backs / UPSERTs into the data stores).
//!
//! # The running context (~100k token cap)
//!
//! Gaia's context starts empty and is built up over consecutive turns, kept
//! summarized/compacted under [`CONTEXT_TOKEN_CAP`] tokens. It is composed of
//! the sections in [`context_budget`], each with its own token budget:
//!
//! | Section | Budget | Notes |
//! |---|---|---|
//! | Identity (Gaia Context) | 2k | Who/where she is; mostly static |
//! | Analysis | 1k | Latest emotion / truthfulness / intention read |
//! | Facts | 15k | Durable learned facts (UPSERTed each turn) |
//! | Conversation history | 20k | Compacted prior turns |
//! | Response Data Context | 50k | This turn's pull results — **Call 2 only** |
//! | Carry-over (newContext) | 12k | Call 1's context + output, compressed |
//!
//! Two mechanisms keep the context bounded: a per-turn **decay** (Call 1
//! compresses its full context + output to ~[`CONTEXT_COMPRESSION_RATIO`] of its
//! size into `newContext.json`, which is all Call 2 inherits of Call 1) and a
//! hard **compaction** whenever the assembled context would exceed the cap.
//!
//! Note the asymmetry: the **Response Data Context is blank for Call 1** and is
//! only populated (from the executed pull actions) on the way into Call 2.
//!
//! # This file today
//!
//! The blocks below are still an interactive **skeleton**: each block logs its
//! intended behaviour and waits for Enter so we can walk the flow by hand. The
//! `[intended]` / `[TODO]` lines printed per block (see [`narrate`]) describe the
//! real behaviour we are building toward; the `TODO`s mark where live
//! implementation (LLM calls, action execution, storage writes) will be wired in.
//!
//! The loop runs until the user presses Esc (or input reaches end-of-file).

// Each module holds exactly one primary type plus its tests.
mod actions;
mod auth;
mod base64;
mod connection;
mod cosmos;
mod diary;
mod engine;
mod executor;
mod flow;
mod http_request;
mod http_response;
mod llm;
mod prompt;
mod search_history;
mod server;
mod sha1;
mod storage;
mod test_data_retrieval;
mod web_search;
mod websocket;

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

/// The Esc key. When this control character appears in a line of input we treat
/// it as the signal to stop the program.
///
/// Note: a portable, std-only console reads input a line at a time, so the user
/// presses Esc and then Enter. Detecting a bare single Esc keystroke would
/// require putting the terminal into raw mode (e.g. via the `crossterm` crate),
/// which this skeleton intentionally avoids to stay dependency-free.
const ESC: char = '\u{1b}';

/// Hard upper bound on the running context, in approximate tokens. When the
/// assembled context would exceed this, older sections are summarized/compacted
/// to fit. See [`context_budget`] for how the cap is divided across sections.
const CONTEXT_TOKEN_CAP: usize = 100_000;

/// Compression ratio applied by LLM Call 1 when it summarizes its full
/// context + output into `newContext.json` for LLM Call 2. ~0.61 keeps the
/// carry-over compact while preserving *what Call 1 searched for and why*, so
/// Call 2 inherits Call 1's reasoning without its full bulk.
const CONTEXT_COMPRESSION_RATIO: f32 = 0.61;

/// Per-section token budgets for Gaia's running context.
///
/// The budgets sum to [`CONTEXT_TOKEN_CAP`]. They are a *target layout*: not
/// every section is present in every pass (notably `RESPONSE_DATA_CONTEXT` is
/// populated only for LLM Call 2). `main` logs these numbers as it narrates the
/// flow so the intended sizing stays visible while we build the real assembler.
mod context_budget {
    /// Identity: who and where Gaia is. Mostly static.
    pub const IDENTITY: usize = 2_000;
    /// Latest analysis of the user (emotion / truthfulness / intention).
    pub const ANALYSIS: usize = 1_000;
    /// Durable facts learned about the user and the world.
    pub const FACTS: usize = 15_000;
    /// Compacted summary of prior conversation turns.
    pub const CONVERSATION_HISTORY: usize = 20_000;
    /// This turn's pull-action results. Present for LLM Call 2 only.
    pub const RESPONSE_DATA_CONTEXT: usize = 50_000;
    /// Carry-over from LLM Call 1 (`newContext.json`), already compressed.
    pub const CARRY_OVER: usize = 12_000;
}

/// Program entry point.
///
/// Returns [`ExitCode::SUCCESS`] on a clean exit and [`ExitCode::FAILURE`] if
/// the program cannot read from or write to its standard streams. We return an
/// `ExitCode` rather than panicking so failures surface as a normal, testable
/// exit status.
fn main() -> ExitCode {
    // --- 0a. Data-retrieval self-test mode (opt-in via subcommand) ---------
    // `gaia-robot test-data-retrieval` runs the five-question pull-pass probe
    // (LLM Call 1 -> actions.json -> Cosmos + Brave) and exits non-zero on any
    // failure, so it doubles as an on-demand check and a hard CI deploy gate.
    // It is checked first so it never collides with server or console mode.
    if wants_data_retrieval_test() {
        return run_data_retrieval_test();
    }

    // --- 0. HTTP server mode (opt-in) --------------------------------------
    // When a listen address is configured (`GAIA_HTTP_ADDR` or `GAIA_HTTP_PORT`),
    // run the backend server instead of the interactive console. This is what the
    // PWA front end talks to: it kicks off the same two-pass thought sequence via
    // `engine::Engine` and returns the reply over HTTP/WebSocket. Without those
    // vars the program keeps its default console behaviour, so the CLI tests and
    // the hand-driven walk-through are unchanged.
    if let Some(addr) = server::http_addr_from_env() {
        return run_server(&addr);
    }

    // --- 1. Set up the flow steps and the I/O streams ----------------------
    // The flow module owns the eleven blocks; `main` just drives them in order.
    let steps = flow::steps();

    // Lock stdin/stdout once up front for efficient line-by-line interaction.
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    // --- 2. Print the banner -----------------------------------------------
    // `writeln!` can fail if stdout is closed; treat that as a fatal error.
    if writeln!(
        output,
        "Gaia program flow. Type 'quit' (or 'exit') then Enter at any prompt to quit.",
    )
    .is_err()
    {
        return ExitCode::FAILURE;
    }

    // --- 3. Decide whether to run real LLM calls (dev/local mode) ----------
    // In dev/local mode (`GAIA_MODE=dev`) the two "LLM Call" blocks make a real
    // request to GitHub Models. Otherwise the program keeps its original
    // skeleton behaviour and simply logs each block.
    let llm_client = match llm::LlmClient::from_env() {
        Ok(Some(client)) => {
            if writeln!(
                output,
                "Dev/local LLM mode enabled: calls go to {} ({}).",
                client.endpoint(),
                client.model(),
            )
            .is_err()
            {
                return ExitCode::FAILURE;
            }
            Some(client)
        }
        Ok(None) => None,
        Err(err) => {
            // Dev mode was requested but isn't configured; warn and continue in
            // skeleton mode rather than failing the whole program.
            if writeln!(
                output,
                "LLM dev mode requested but not configured: {err}. Running in skeleton mode.",
            )
            .is_err()
            {
                return ExitCode::FAILURE;
            }
            None
        }
    };

    // In dev/local mode every read and write is scoped to a single GitHub user.
    // Resolve that id from `GAIA_USER_ID` (env or .env, default `threadkeeper`).
    // Both tailored prompts (Call 1 and Call 2) take this id directly and scope
    // everything they read, write, and say to this user only. In skeleton mode
    // there is no user id.
    let dev_user_id = if llm_client.is_some() {
        llm::dev_user_id()
    } else {
        String::new()
    };
    if !dev_user_id.is_empty()
        && writeln!(
            output,
            "User isolation: scoped to user_id \"{dev_user_id}\"."
        )
        .is_err()
    {
        return ExitCode::FAILURE;
    }

    // The web-search applet (Brave Search API). It is optional: only built when
    // a `BRAVE_SEARCH_API_KEY` is configured (env or infra/.env). When present,
    // LLM Call 1's `actions.json` Web action runs a real search and the query +
    // results are appended to the Gaia Search History audit log below. In
    // skeleton mode (no dev LLM client) we leave web search off entirely.
    let web_search_client = if llm_client.is_some() {
        web_search::BraveClient::from_env()
    } else {
        None
    };
    if let Some(brave) = &web_search_client {
        if writeln!(
            output,
            "Web search enabled: Brave Search API at {}.",
            brave.endpoint(),
        )
        .is_err()
        {
            return ExitCode::FAILURE;
        }
    }

    // Append-only audit log of every web search Gaia runs this session. Logged
    // only — never embedded or indexed (see search_history::SearchHistory).
    let mut search_history = search_history::SearchHistory::default();

    // --- 4. Run the program-flow loop --------------------------------------
    // Each outer iteration walks all eleven blocks once. We reuse a single
    // input buffer across reads to avoid needless allocations. The most recent
    // user input is threaded into the LLM steps within the same pass.
    let mut line = String::new();
    let mut last_user_input = String::new();
    // Gaia's running conversation history. It starts empty and is intended to
    // accumulate compacted prior turns; LLM Call 1 receives it as context. We
    // do not build it yet, so for now it is threaded through empty.
    let conversation_history = String::new();
    // The Response Data Context handed to LLM Call 2. It is assembled by
    // executing LLM Call 1's read-only actions against Cosmos (see
    // `run_pull_actions` below) and is empty until that first pull pass runs.
    let mut response_data_context = String::new();

    // The Cosmos client that executes LLM Call 1's authored read-only queries.
    // Built only in dev/local mode and only when Cosmos is configured; without
    // it the pull pass is skipped and Call 2 simply receives an empty context,
    // exactly as a degraded turn would. Configuration problems are non-fatal.
    let cosmos_client = if llm_client.is_some() {
        match cosmos::CosmosClient::from_env() {
            Ok(Some(client)) => {
                if writeln!(
                    output,
                    "Cosmos enabled: pull queries run against {}.",
                    client.endpoint()
                )
                .is_err()
                {
                    return ExitCode::FAILURE;
                }
                Some(client)
            }
            Ok(None) => None,
            Err(err) => {
                if writeln!(
                    output,
                    "Cosmos requested but not configured: {err}. Pull pass disabled.",
                )
                .is_err()
                {
                    return ExitCode::FAILURE;
                }
                None
            }
        }
    } else {
        None
    };

    'flow: loop {
        for (index, step) in steps.iter().enumerate() {
            // Log the block's exact description, mirroring the diagram, with rainbow color coding.
            let color = step_color_code(index);
            let reset = "\x1b[0m";
            if writeln!(
                output,
                "\n{}{}{} === {} ==={}",
                color,
                " ".repeat(0),
                color,
                step.title(),
                reset
            )
            .is_err()
                || writeln!(output, "{}{}{}", color, step.description(), reset).is_err()
            {
                return ExitCode::FAILURE;
            }

            // Narrate the block's *intended* runtime behaviour (with TODOs and
            // the relevant token budgets) so the walk-through documents the real
            // design we are building toward, not just the diagram text.
            if narrate(index, &mut output).is_err() {
                return ExitCode::FAILURE;
            }

            // In dev/local mode the two "LLM Call" blocks make a live model
            // request using the Gaia context and the latest user input.
            if let Some(client) = &llm_client {
                if llm::is_llm_call(step.title()) {
                    // Form the message pair for this specific call. LLM Call 1
                    // (the pull pass) gets the tailored research prompt; LLM
                    // Call 2 (the push pass) gets the tailored answer prompt and
                    // the assembled Response Data Context.
                    let (system, user) = if step.title() == "LLM Call 1" {
                        let formed = prompt::Call1Prompt::build(
                            &dev_user_id,
                            &last_user_input,
                            &conversation_history,
                            &prompt::now_rfc3339(),
                        );
                        (formed.system, formed.user)
                    } else {
                        // The Response Data Context is assembled by executing
                        // Call 1's actions; that executor is not wired yet, so we
                        // pass an empty context and Call2Prompt fills a clear
                        // placeholder. See research/ResponseDataContext.md for the
                        // shape this will carry once the executor runs.
                        let formed = prompt::Call2Prompt::build(
                            &dev_user_id,
                            &last_user_input,
                            &response_data_context,
                            &prompt::now_rfc3339(),
                        );
                        (formed.system, formed.user)
                    };

                    // Print the full request first: the complete context window
                    // plus the (currently empty) set of attached tools / MCP
                    // servers / skills, so nothing sent to the model is hidden.
                    // Print the context window in black text on white background.
                    if writeln!(output, "    --- {} request being sent ---", step.title()).is_err()
                    {
                        return ExitCode::FAILURE;
                    }
                    let preview = client.request_preview(&system, &user);
                    // Print the context window in black text on a white
                    // background. We style each line individually (see
                    // print_black_on_white) because terminals reset the
                    // background at every newline, so a single multiline
                    // wrapper would only highlight the first line.
                    if print_black_on_white(&mut output, &preview).is_err() {
                        return ExitCode::FAILURE;
                    }

                    let completion = client.complete(&system, &user);
                    let rendered = match &completion {
                        Ok(reply) => format!("[{} via {}]\n{reply}", step.title(), client.model()),
                        Err(err) => format!("[{} failed] {err}", step.title()),
                    };
                    if writeln!(output, "{rendered}").is_err() {
                        return ExitCode::FAILURE;
                    }

                    // After LLM Call 1 (the pull pass) succeeds, execute the
                    // read-only Cosmos queries it authored and assemble the
                    // Response Data Context that LLM Call 2 will consume. Web
                    // actions are skipped here — they are run by the Brave applet
                    // in block 3 below. The pull pass only runs when a Cosmos
                    // client is configured; otherwise the context stays empty.
                    if step.title() == "LLM Call 1" {
                        if let (Ok(reply), Some(cosmos)) = (&completion, &cosmos_client) {
                            let (context, log) = run_pull_actions(cosmos, reply);
                            if writeln!(output, "{log}").is_err() {
                                return ExitCode::FAILURE;
                            }
                            response_data_context = context;
                        }
                    }
                }
            }

            // Block 3 is LLM Call 1's `actions.json` — the read-only GET actions,
            // whose `Web` entry is Gaia's web search. When a Brave client is
            // configured we actually run a search for this turn's input, print
            // the results, and append them to the Search History audit log. The
            // query is the user's sentence today; once the executor parses
            // actions.json it will use the model's chosen query string instead.
            if index == 3 {
                if let Some(brave) = &web_search_client {
                    let query = last_user_input.trim();
                    if !query.is_empty()
                        && run_web_search(brave, query, &mut search_history, &mut output).is_err()
                    {
                        return ExitCode::FAILURE;
                    }
                }
            }

            // The first block (User) prompts for input; every other block just
            // waits for Enter before moving to the next part of the flow.
            let prompt = if index == 0 {
                "Your input> "
            } else {
                "Press Enter to continue (type 'quit' to exit)> "
            };
            if write!(output, "{prompt}").is_err() || output.flush().is_err() {
                return ExitCode::FAILURE;
            }

            // Read one line of input to advance the flow.
            line.clear();
            match input.read_line(&mut line) {
                // End of input (e.g. piped file or Ctrl-Z/Ctrl-D): exit cleanly.
                Ok(0) => break 'flow,
                Ok(_) => {}
                // Could not read input at all: nothing sensible to do but fail.
                Err(_) => return ExitCode::FAILURE,
            }

            // The User block's line becomes the input threaded into the LLM
            // steps later in this same pass.
            if index == 0 {
                last_user_input = line.trim().to_string();
            }

            // Quitting works two ways. A bare Esc keystroke is unreliable in a
            // line-buffered terminal (the shell's line editor swallows it), so
            // we also accept a typed quit word. Either ends the whole loop, not
            // just this block.
            let trimmed = line.trim();
            if line.contains(ESC) || is_quit_word(trimmed) {
                break 'flow;
            }
        }
    }

    // --- 5. Exit -----------------------------------------------------------
    if writeln!(output, "\nGoodbye!").is_err() {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Return `true` when the program was invoked as the data-retrieval self-test.
///
/// Triggered by a `test-data-retrieval` (or `TestDataRetrieval`) argument so the
/// probe can be launched as `gaia-robot test-data-retrieval`. The comparison is
/// case-insensitive so the PowerShell wrapper name and the CLI form both work.
fn wants_data_retrieval_test() -> bool {
    std::env::args().skip(1).any(|arg| {
        arg.eq_ignore_ascii_case("test-data-retrieval")
            || arg.eq_ignore_ascii_case("testdataretrieval")
    })
}

/// Run the data-retrieval self-test and map its result to an exit code.
///
/// Builds the probe from the environment (the same dev/local configuration the
/// rest of the program uses), runs all five questions, and returns
/// [`ExitCode::SUCCESS`] only when every question passed. A configuration
/// problem (no model, etc.) or any retrieval failure returns
/// [`ExitCode::FAILURE`] so CI halts before deploying.
fn run_data_retrieval_test() -> ExitCode {
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let probe = match test_data_retrieval::DataRetrievalProbe::from_env() {
        Ok(probe) => probe,
        Err(err) => {
            // A self-test that cannot even start has validated nothing, so we
            // fail closed rather than reporting a misleading success.
            let _ = writeln!(out, "data-retrieval self-test cannot run: {err}");
            return ExitCode::FAILURE;
        }
    };

    match probe.run(&mut out) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        // An I/O error writing the report is itself a failure.
        Err(_) => ExitCode::FAILURE,
    }
}

/// Run the backend HTTP/WebSocket server, blocking until it stops.
///
/// Builds the turn [`engine::Engine`] from the same environment the console app
/// uses (so dev mode, the model, and web search are configured identically),
/// then serves the PWA front end on `addr`. Returns [`ExitCode::FAILURE`] only if
/// the listener cannot be bound; per-connection errors are isolated inside the
/// server and never bring it down.
fn run_server(addr: &str) -> ExitCode {
    // Build the engine and surface any non-fatal configuration warning (e.g. dev
    // mode requested but the model is misconfigured) before we start serving.
    let (engine, warning) = engine::Engine::from_env();
    if let Some(warning) = warning {
        eprintln!("engine configuration warning: {warning}");
    }

    // Auth manager: live Google sign-in when GOOGLE_CLIENT_ID is set,
    // otherwise dev-auth (Bearer dev:<name>).
    let auth = auth::Auth::from_env();

    let server = server::Server::new(engine, auth);
    match server.serve(addr) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("failed to start Gaia backend on {addr}: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Run one web search through the Brave client, print the results, and append
/// them to the Search History audit log.
///
/// This is the live implementation of LLM Call 1's `Web` GET action. A failed
/// search is *non-fatal*: we print the error (so the walk-through shows what
/// went wrong, e.g. a missing/expired key) and still log the attempt with no
/// results, exactly as a degraded turn would. Only an I/O error writing to
/// `out` is propagated, matching the rest of `main`.
fn run_web_search(
    client: &web_search::BraveClient,
    query: &str,
    history: &mut search_history::SearchHistory,
    out: &mut impl Write,
) -> io::Result<()> {
    writeln!(out, "    --- web search (Brave) ---")?;
    writeln!(out, "    query   : {query}")?;

    // Default result count (the executor will pass the action's `top` later).
    let results = match client.search(query, 0) {
        Ok(results) => results,
        Err(err) => {
            // Degrade gracefully: report the failure and log an empty result set
            // so the audit trail still records that a search was attempted.
            writeln!(out, "    error   : {err}")?;
            Vec::new()
        }
    };

    if results.is_empty() {
        writeln!(out, "    results : (none)")?;
    } else {
        writeln!(out, "    results : {}", results.len())?;
        for (rank, result) in results.iter().enumerate() {
            writeln!(
                out,
                "      {}. {} \u{2014} {}",
                rank + 1,
                result.title,
                result.url
            )?;
        }
    }

    // Append to the append-only audit log (logged only, never embedded).
    history.record(prompt::now_rfc3339(), query, results);
    writeln!(
        out,
        "    logged  : Search History now holds {} entr{}",
        history.len(),
        if history.len() == 1 { "y" } else { "ies" },
    )?;

    Ok(())
}

/// Approximate character budget for the Response Data Context.
///
/// Token budgets are easiest to enforce at the character level (~4 chars per
/// token is a safe over-estimate). We cap the assembled pull results at this
/// many characters so the context handed to LLM Call 2 stays well within its
/// [`context_budget::RESPONSE_DATA_CONTEXT`] token slice (requirement 3: the
/// returned data must be small enough for Call 2 to consume comfortably).
const RESPONSE_DATA_CONTEXT_CHAR_CAP: usize = context_budget::RESPONSE_DATA_CONTEXT * 4;

/// Maximum characters of any single record's payload kept in the context.
///
/// Individual records are trimmed so one chatty document cannot crowd out the
/// others; the model only needs a representative snippet to reason over.
const RECORD_SNIPPET_CHAR_CAP: usize = 600;

/// Execute LLM Call 1's authored read-only queries and assemble the Response
/// Data Context for LLM Call 2.
///
/// `call1_reply` is the raw model output for Call 1 — a JSON array whose first
/// element is the `actions.json` document. We parse it, drop the `Web` action
/// (run separately by the Brave applet), run the remaining queries against
/// Cosmos, and fold the results into a compact, bounded context string.
///
/// Returns `(context, log)`: the context handed to Call 2, and a human-readable
/// log of what was planned and retrieved (printed during the walk-through). A
/// failure to parse or a per-action error is captured in the log and never
/// aborts the turn — a degraded pull simply yields a smaller context.
fn run_pull_actions(cosmos: &cosmos::CosmosClient, call1_reply: &str) -> (String, String) {
    let mut log = String::from("    --- executing LLM Call 1 pull queries against Cosmos ---\n");

    // Parse the actions.json document out of Call 1's output.
    let actions = match parse_actions_from_call1(call1_reply) {
        Some(actions) => actions,
        None => {
            log.push_str("    could not parse actions.json from Call 1 output; skipping pull.\n");
            return (String::new(), log);
        }
    };

    // Keep only the Cosmos-backed actions; the Web action is handled elsewhere.
    let cosmos_actions = cosmos_actions_of(&actions);
    if cosmos_actions.is_empty() {
        log.push_str("    no Cosmos pull actions in this turn.\n");
        return (String::new(), log);
    }

    // Show the exact SQL each action will run (the authored `query` field).
    log.push_str(&describe_authored_queries(&cosmos_actions));

    // Run the queries. The executor truncates each result to the action's
    // `top`, and the Cosmos client caps items per request, so the volume is
    // bounded before we ever assemble the context.
    let filtered = actions::ActionsFile {
        version: actions.version.clone(),
        session: actions.session.clone(),
        actions: cosmos_actions,
    };
    let outcomes = executor::Executor::new(cosmos).run(&filtered);

    // Fold the outcomes into a compact, bounded context plus a run log.
    let (context, outcome_log) = summarize_pull_outcomes(&outcomes);
    log.push_str(&outcome_log);
    (context, log)
}

/// Select the Cosmos-backed actions from an action plan.
///
/// The `Web` action is dropped — it is served by the Brave applet, not Cosmos —
/// and every remaining action is returned in order.
fn cosmos_actions_of(actions: &actions::ActionsFile) -> Vec<actions::ActionPlan> {
    actions
        .actions
        .iter()
        .filter(|action| !action.target.eq_ignore_ascii_case("Web"))
        .cloned()
        .collect()
}

/// Render the exact SQL each action will run as an indented log block.
///
/// Each line shows the action id, its target container, and either the planned
/// SQL or the reason no query could be planned, so the authored `query` field
/// is visible during the walk-through.
fn describe_authored_queries(actions: &[actions::ActionPlan]) -> String {
    let mut log = String::new();
    for action in actions {
        match executor::plan_for(action) {
            Ok(planned) => {
                log.push_str(&format!(
                    "    [{}] {} :: {}\n",
                    action.id, action.target, planned.sql
                ));
            }
            Err(err) => {
                log.push_str(&format!(
                    "    [{}] {} :: (no query: {err})\n",
                    action.id, action.target
                ));
            }
        }
    }
    log
}

/// Fold executed action outcomes into `(context, log)`.
///
/// `context` is the bounded Response Data Context for LLM Call 2 — one block per
/// action with each record trimmed to a snippet, the whole thing capped to the
/// budget (requirement 3). `log` mirrors what happened for the walk-through.
fn summarize_pull_outcomes(outcomes: &[executor::ActionOutcome]) -> (String, String) {
    let mut context = String::new();
    let mut log = String::new();
    for outcome in outcomes {
        match &outcome.result {
            Ok(records) => {
                log.push_str(&format!(
                    "    [{}] returned {} record(s)\n",
                    outcome.id,
                    records.len()
                ));
                context.push_str(&format!(
                    "## {} ({} record(s))\n",
                    outcome.id,
                    records.len()
                ));
                for record in records {
                    let snippet = truncate_chars(record.data.trim(), RECORD_SNIPPET_CHAR_CAP);
                    context.push_str(&format!(
                        "- {} [{}]: {}\n",
                        record.business_key(),
                        record.date,
                        snippet,
                    ));
                }
            }
            Err(err) => {
                log.push_str(&format!("    [{}] error: {err}\n", outcome.id));
                context.push_str(&format!("## {} (error)\n{err}\n", outcome.id));
            }
        }
    }
    // Final guard: keep the whole context within its budget (requirement 3).
    (
        truncate_chars(&context, RESPONSE_DATA_CONTEXT_CHAR_CAP),
        log,
    )
}

/// Extract the `actions.json` document from LLM Call 1's raw reply.
///
/// Call 1 emits a JSON array of four documents; the first is `actions.json`.
/// This delegates to the shared [`actions::parse_call1_actions`] parser so the
/// console pull pass and the data-retrieval self-test agree on exactly how Call
/// 1 output is interpreted. Returns `None` if anything about that shape is off,
/// so the caller can degrade gracefully rather than fail the turn.
fn parse_actions_from_call1(call1_reply: &str) -> Option<actions::ActionsFile> {
    actions::parse_call1_actions(call1_reply)
}

/// Truncate `text` to at most `max` characters, appending an ellipsis marker
/// when anything was dropped.
///
/// Counting and slicing by `char` keeps us on UTF-8 boundaries so we never panic
/// on multi-byte text.
fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}…(truncated)")
}

/// Narrate one block's *intended* runtime behaviour to `out`.
///
/// This is documentation-as-output: every block prints the real behaviour we are
/// building toward (`[intended]`), the work still to wire up (`[TODO]`), and any
/// relevant token budgets. Keeping it next to `main` lets a reader follow the
/// whole intended pipeline — the pull/push split, how the context is assembled
/// and bounded, and where each `*.json` document flows — by reading this file
/// top-to-bottom.
///
/// `index` is the block's position in [`flow::steps`] (0 = User, 10 =
/// response.json). It returns any I/O error from writing to `out` so the caller
/// can treat a closed stream as fatal, exactly like the rest of `main`.
fn narrate(index: usize, out: &mut impl Write) -> io::Result<()> {
    use context_budget as budget;

    match index {
        // 0. User — the only block that *collects* input.
        0 => {
            writeln!(
                out,
                "    [intended] Capture the user's raw utterance; this is the sole input block.",
            )?;
            writeln!(
                out,
                "    [TODO] Sanitize the input and stamp it with user_id + timestamp for user isolation.",
            )?;
        }
        // 1. Gaia Context — assemble LLM Call 1's (pull) context.
        1 => {
            writeln!(
                out,
                "    [intended] Assemble LLM Call 1's context from the running sections \
                 (the Response Data Context is intentionally absent here):",
            )?;
            writeln!(
                out,
                "                identity {}t + analysis {}t + facts {}t + history {}t + carry-over {}t",
                budget::IDENTITY,
                budget::ANALYSIS,
                budget::FACTS,
                budget::CONVERSATION_HISTORY,
                budget::CARRY_OVER,
            )?;
            writeln!(
                out,
                "    [intended] Keep the assembled context under the {CONTEXT_TOKEN_CAP}-token cap; \
                 hard-compact older sections when it would overflow.",
            )?;
            writeln!(
                out,
                "    [TODO] Build the context struct, enforce per-section budgets, and compact at the cap.",
            )?;
        }
        // 2. LLM Call 1 — the pull pass.
        2 => {
            writeln!(
                out,
                "    [intended] PULL pass: feed the assembled context + the user's sentence to the model.",
            )?;
            writeln!(
                out,
                "    [intended] Emit four documents into an in-memory structure: \
                 actions.json, analysis.json, facts.json, newContext.json.",
            )?;
            writeln!(
                out,
                "    [intended] connection is evaluated later, in LLM Call 2 (the push pass), not here.",
            )?;
            writeln!(
                out,
                "    [TODO] Invoke llm::LlmClient (GAIA_MODE=dev) and parse the four documents.",
            )?;
        }
        // 3. actions.json (Call 1) — read / pull actions only.
        3 => {
            writeln!(
                out,
                "    [intended] READ / non-side-effecting actions only: web search and semantic/logical index queries.",
            )?;
            writeln!(
                out,
                "    [intended] Run each via its applet and collect the results into the Response Data Context.",
            )?;
            writeln!(
                out,
                "    [live] Web search is wired: when BRAVE_SEARCH_API_KEY is set, the Web action calls the Brave \
                 Search API (web_search::BraveClient) and logs the query + results to the Gaia Search History \
                 (search_history::SearchHistory) for audit only — logging, no embedding.",
            )?;
            writeln!(
                out,
                "    [TODO] Deserialize into actions::ActionsFile, validate user isolation, then dispatch the index reads.",
            )?;
        }
        // 4. analysis.json — Call 1's read of the user.
        4 => {
            writeln!(
                out,
                "    [intended] Capture emotion / truthfulness / intention about the user; fold into the Analysis section.",
            )?;
            writeln!(
                out,
                "    [TODO] Parse analysis.json and update the Analysis section ({}-token budget).",
                budget::ANALYSIS,
            )?;
        }
        // 5. facts.json — durable facts learned this turn.
        5 => {
            writeln!(
                out,
                "    [intended] Extract durable facts and UPSERT them into the Facts section ({}-token budget).",
                budget::FACTS,
            )?;
            writeln!(
                out,
                "    [TODO] Merge facts through storage:: tables using KB UPSERT semantics.",
            )?;
        }
        // 6. newContext.json — the compressed carry-over to Call 2.
        6 => {
            writeln!(
                out,
                "    [intended] Compress Call 1's full context + output to ~{:.0}% as newContext.json — \
                 the only part of Call 1 that Call 2 inherits.",
                CONTEXT_COMPRESSION_RATIO * 100.0,
            )?;
            writeln!(
                out,
                "    [intended] Preserve WHAT Call 1 searched for and WHY, within the {}-token carry-over budget.",
                budget::CARRY_OVER,
            )?;
            writeln!(
                out,
                "    [TODO] Produce the summary and store it as the carry-over for Call 2 and the next turn.",
            )?;
        }
        // 7. Response Data Context — built between the calls, for Call 2 only.
        7 => {
            writeln!(
                out,
                "    [intended] In-memory store holding the results of the executed pull actions; built between the two calls.",
            )?;
            writeln!(
                out,
                "    [intended] Present for LLM Call 2 ONLY, within a {}-token budget.",
                budget::RESPONSE_DATA_CONTEXT,
            )?;
            writeln!(
                out,
                "    [TODO] Populate it from the action results and trim to budget before Call 2.",
            )?;
        }
        // 8. LLM Call 2 — the push pass.
        8 => {
            writeln!(
                out,
                "    [intended] PUSH pass: feed identity + newContext (carry-over) + Response Data Context to the model.",
            )?;
            writeln!(
                out,
                "    [intended] Answer the user (response.json) AND emit side-effecting actions (actions.json).",
            )?;
            writeln!(
                out,
                "    [intended] connection: judge whether the input grows or weakens friendship and pick a signed \
                 change to Gaia's per-user \"emotional bank account\" (connection.json), then send it on for action.",
            )?;
            writeln!(
                out,
                "    [TODO] Invoke llm::LlmClient and parse response.json + connection.json + the push actions.",
            )?;
        }
        // 9. actions.json (Call 2) — side-effecting / push actions.
        9 => {
            writeln!(
                out,
                "    [intended] SIDE-EFFECTING actions: send WhatsApp/Push, actuate, emote, and write-backs (UPSERT) to the stores.",
            )?;
            writeln!(
                out,
                "    [intended] Post Call 2's connection change to the Gaia Connections ledger (connection::ConnectionLedger), \
                 keyed by (entity, timestamp), recording change + previous/new balance + notes.",
            )?;
            writeln!(
                out,
                "    [intended] Not safely repeatable — guard them and require confirmation where appropriate.",
            )?;
            writeln!(
                out,
                "    [TODO] Execute through storage:: tables, the connection ledger, and the actuator applets.",
            )?;
        }
        // 10. response.json — deliver the answer and close the loop.
        10 => {
            writeln!(
                out,
                "    [intended] Deliver the final answer to the user, closing this turn.",
            )?;
            writeln!(
                out,
                "    [intended] Carry-over decays (~{:.0}%) into the next turn; hard-compact at {CONTEXT_TOKEN_CAP} tokens.",
                CONTEXT_COMPRESSION_RATIO * 100.0,
            )?;
            writeln!(
                out,
                "    [intended] Write a diary note for this session to the Gaia Diary (diary::Diary), keyed by (wing, timestamp).",
            )?;
            writeln!(
                out,
                "    [TODO] Emit response.json, persist the turn, and feed the decayed context into the next loop.",
            )?;
        }
        _ => {}
    }

    Ok(())
}

/// Print text line-by-line in black on white background.
///
/// Some terminals handle multiline ANSI wrappers inconsistently. Applying the
/// style per line makes the context-window formatting deterministic.
fn print_black_on_white(out: &mut impl Write, text: &str) -> io::Result<()> {
    for line in text.lines() {
        writeln!(out, "\x1b[30;47m{line}\x1b[0m")?;
    }

    // Preserve a trailing blank line if the original text ended with '\n'.
    if text.ends_with('\n') {
        writeln!(out, "\x1b[30;47m\x1b[0m")?;
    }

    Ok(())
}

/// Return `true` if a line of input is a typed request to quit the flow.
///
/// A bare Esc keystroke is unreliable in a line-buffered terminal because the
/// shell's line editor swallows it before the program ever sees the control
/// character. Accepting a typed word gives the user a quit path that always
/// works regardless of terminal mode. We compare against a small set of common
/// quit words, case-insensitively, after trimming surrounding whitespace (the
/// caller passes an already-trimmed slice, but we stay robust).
fn is_quit_word(line: &str) -> bool {
    matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "quit" | "exit" | "q"
    )
}

/// Return the ANSI color code for a flow step's output, cycling through rainbow colors.
///
/// The eleven steps are colored as follows:
/// 0. User — Red
/// 1. Gaia Context — Orange (bright yellow)
/// 2. LLM Call 1 — Yellow
/// 3. actions.json (Call 1) — Green
/// 4. analysis.json — Cyan
/// 5. facts.json — Blue
/// 6. newContext.json — Magenta
/// 7. Response Data Context — Bright Red
/// 8. LLM Call 2 — Bright Green
/// 9. actions.json (Call 2) — Bright Cyan
/// 10. response.json — Bright Magenta
fn step_color_code(index: usize) -> &'static str {
    match index {
        0 => "\x1b[31m",  // Red
        1 => "\x1b[93m",  // Bright Yellow (Orange)
        2 => "\x1b[33m",  // Yellow
        3 => "\x1b[32m",  // Green
        4 => "\x1b[36m",  // Cyan
        5 => "\x1b[34m",  // Blue
        6 => "\x1b[35m",  // Magenta
        7 => "\x1b[91m",  // Bright Red
        8 => "\x1b[92m",  // Bright Green
        9 => "\x1b[96m",  // Bright Cyan
        10 => "\x1b[95m", // Bright Magenta
        _ => "\x1b[0m",   // Default (reset) for any other index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_quit_words_case_insensitively() {
        for word in ["quit", "QUIT", "Exit", "  q  ", "eXiT"] {
            assert!(is_quit_word(word), "{word:?} should be a quit word");
        }
    }

    #[test]
    fn ordinary_input_is_not_a_quit_word() {
        for word in ["", "hello", "quitter", "exits", "question"] {
            assert!(!is_quit_word(word), "{word:?} should not be a quit word");
        }
    }

    #[test]
    fn narrate_describes_every_flow_block() {
        // Blocks 0..=10 each emit at least one `[intended]` narration line; the
        // out-of-range index is a clean no-op. We capture into a buffer so the
        // narration is exercised without a real terminal.
        for index in 0..=10 {
            let mut buffer = Vec::new();
            narrate(index, &mut buffer).expect("narration writes to the buffer");
            let text = String::from_utf8(buffer).expect("narration is valid UTF-8");
            assert!(
                text.contains("[intended]"),
                "block {index} should narrate intended behaviour"
            );
        }

        // Any index past the known blocks narrates nothing and still succeeds.
        let mut buffer = Vec::new();
        narrate(99, &mut buffer).expect("unknown block is a no-op");
        assert!(buffer.is_empty());
    }

    #[test]
    fn step_color_code_is_distinct_per_block_and_resets_otherwise() {
        // Each of the eleven blocks has its own colour; unknown indices reset.
        let codes: Vec<&str> = (0..=10).map(step_color_code).collect();
        let unique: std::collections::BTreeSet<&str> = codes.iter().copied().collect();
        assert_eq!(unique.len(), 11, "every block colour should be distinct");
        assert_eq!(step_color_code(42), "\x1b[0m");
    }

    #[test]
    fn print_black_on_white_styles_each_line_and_keeps_trailing_blank() {
        let mut buffer = Vec::new();
        print_black_on_white(&mut buffer, "one\ntwo\n").expect("write to buffer");
        let text = String::from_utf8(buffer).expect("styled output is UTF-8");
        // Both content lines are styled.
        assert!(text.contains("\x1b[30;47mone\x1b[0m"));
        assert!(text.contains("\x1b[30;47mtwo\x1b[0m"));
        // The trailing newline preserves one extra styled blank line.
        assert_eq!(text.matches("\x1b[0m").count(), 3);
    }

    #[test]
    fn truncate_chars_keeps_short_text_verbatim() {
        // Text within the limit is returned unchanged and unmarked.
        assert_eq!(truncate_chars("hello", 10), "hello");
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_trims_long_text_on_char_boundaries() {
        // Over-long text is cut to `max` chars with a marker appended. Using a
        // multi-byte string proves we never split inside a UTF-8 scalar.
        let trimmed = truncate_chars("élan vital", 4);
        assert_eq!(trimmed, "élan…(truncated)");
        assert_eq!(trimmed.chars().take(4).collect::<String>(), "élan");
    }

    #[test]
    fn parse_actions_from_call1_extracts_the_first_document() {
        // Call 1 emits an array of documents; element 0 is actions.json. We
        // wrap it in code fences and prose to prove the bracket-scan is robust.
        let reply = r#"Here you go:
```json
[
  {"version":"1.0",
   "session":{"user_id":"u1","requested_at":"2026-06-16T12:00:00Z"},
   "actions":[{"id":"q1","kind":"query","target":"GaiaDiary","entity":"threadkeeper",
     "intent":"recent","top":3,
     "query":"SELECT TOP 3 c.id FROM c WHERE c.entity = @pk","filters":{}}]},
  {"analysis":true},{"facts":[]},{"newContext":""}
]
```
thanks"#;

        let parsed = parse_actions_from_call1(reply).expect("should parse actions.json");
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.actions[0].target, "GaiaDiary");
        assert_eq!(
            parsed.actions[0].authored_query(),
            Some("SELECT TOP 3 c.id FROM c WHERE c.entity = @pk")
        );
    }

    #[test]
    fn parse_actions_from_call1_returns_none_for_malformed_output() {
        // No JSON array, or a first element that is not an actions document.
        assert!(parse_actions_from_call1("no json here").is_none());
        assert!(parse_actions_from_call1("[123, 456]").is_none());
    }

    /// Build a minimal action for the pull-helper tests.
    fn action(id: &str, target: &str, query: Option<&str>) -> actions::ActionPlan {
        actions::ActionPlan {
            id: id.to_string(),
            kind: "query".to_string(),
            target: target.to_string(),
            user_id: None,
            entity: Some("threadkeeper".to_string()),
            intent: "recent".to_string(),
            top: 3,
            query: query.map(str::to_string),
            filters: actions::ActionFilters::default(),
        }
    }

    #[test]
    fn cosmos_actions_of_drops_the_web_action() {
        let file = actions::ActionsFile {
            version: "1.0".to_string(),
            session: actions::SessionContext::default(),
            actions: vec![
                action("q1", "Web", None),
                action("q2", "GaiaDiary", None),
                action("q3", "UsersDataLake", None),
            ],
        };

        let kept = cosmos_actions_of(&file);
        let ids: Vec<&str> = kept.iter().map(|a| a.id.as_str()).collect();
        // Web is removed; the Cosmos-backed actions are kept in order.
        assert_eq!(ids, ["q2", "q3"]);
    }

    #[test]
    fn describe_authored_queries_shows_the_sql_per_action() {
        let sql = "SELECT TOP 3 c.id FROM c WHERE c.entity = @pk";
        let log = describe_authored_queries(&[action("q1", "GaiaDiary", Some(sql))]);
        assert!(log.contains("[q1] GaiaDiary ::"));
        assert!(log.contains(sql));
    }

    #[test]
    fn summarize_pull_outcomes_formats_records_and_errors() {
        let record = storage::Record::new(
            "GaiaDiary|threadkeeper|2026-05-10",
            "threadkeeper",
            "",
            "2026-05-10",
            storage::RecordKind::DataLake,
            "  discussed the robot's adventures in nature  ",
            Vec::new(),
        );

        let outcomes = vec![
            executor::ActionOutcome {
                id: "q1".to_string(),
                result: Ok(vec![record]),
            },
            executor::ActionOutcome {
                id: "q2".to_string(),
                result: Err("partition not found".to_string()),
            },
        ];

        let (context, log) = summarize_pull_outcomes(&outcomes);
        // The successful action contributes a header, the record's business key,
        // date, and trimmed snippet.
        assert!(context.contains("## q1 (1 record(s))"));
        assert!(context.contains("threadkeeper [2026-05-10]"));
        assert!(context.contains("discussed the robot's adventures in nature"));
        // The failed action is reported as an error block, not dropped.
        assert!(context.contains("## q2 (error)"));
        assert!(context.contains("partition not found"));
        // The log mirrors both outcomes.
        assert!(log.contains("[q1] returned 1 record(s)"));
        assert!(log.contains("[q2] error: partition not found"));
    }
}
