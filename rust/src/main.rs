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
mod connection;
mod diary;
mod flow;
mod llm;
mod prompt;
mod search_history;
mod storage;

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
        "Gaia program flow. Press Esc then Enter at any prompt to quit.",
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
        && writeln!(output, "User isolation: scoped to user_id \"{dev_user_id}\".").is_err()
    {
        return ExitCode::FAILURE;
    }

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
            // Log the block's exact description, mirroring the diagram.
            if writeln!(output, "\n=== {} ===", step.title()).is_err()
                || writeln!(output, "{}", step.description()).is_err()
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
                    if writeln!(output, "    --- {} request being sent ---", step.title()).is_err()
                    {
                        return ExitCode::FAILURE;
                    }
                    let preview = client.request_preview(&system, &user);
                    if write!(output, "{preview}").is_err() {
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

            // The first block (User) prompts for input; every other block just
            // waits for Enter before moving to the next part of the flow.
            let prompt = if index == 0 {
                "Your input> "
            } else {
                "Press Enter to continue (Esc to quit)> "
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

            // Esc anywhere in the line ends the whole loop, not just this block.
            if line.contains(ESC) {
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
                "    [intended] Log every web search + its results to the Gaia Search History (search_history::SearchHistory) \
                 for audit only \u{2014} logging, no embedding.",
            )?;
            writeln!(
                out,
                "    [TODO] Deserialize into actions::ActionsFile, validate user isolation, then dispatch the reads.",
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
