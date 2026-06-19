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
mod base64;
mod connection;
mod diary;
mod engine;
mod flow;
mod http_request;
mod http_response;
mod llm;
mod prompt;
mod search_history;
mod server;
mod sha1;
mod storage;
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
    // executing LLM Call 1's read-only actions; that executor is not wired yet,
    // so for now it is threaded through empty (Call2Prompt fills a placeholder).
    let response_data_context = String::new();
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

                    let rendered = match client.complete(&system, &user) {
                        Ok(reply) => format!("[{} via {}]\n{reply}", step.title(), client.model()),
                        Err(err) => format!("[{} failed] {err}", step.title()),
                    };
                    if writeln!(output, "{rendered}").is_err() {
                        return ExitCode::FAILURE;
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

    let server = server::Server::new(engine);
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
}
