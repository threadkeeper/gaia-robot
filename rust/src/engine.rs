//! The [`Engine`] type: runs one full Gaia "thought sequence" turn.
//!
//! This is the headless heart of the backend. Where [`main`](crate) walks the
//! eleven flow blocks interactively for a human at a terminal, the `Engine`
//! performs the same two-pass loop for an HTTP/WebSocket caller and returns a
//! structured [`TurnResult`] the front end can render:
//!
//! 1. **LLM Call 1 (pull)** — [`crate::prompt::Call1Prompt`] asks the model what
//!    to research for this turn.
//! 2. **Web search (optional)** — when a Brave client is configured, the user's
//!    sentence is searched and the results become this turn's Response Data
//!    Context (the real retrieved evidence handed to Call 2).
//! 3. **LLM Call 2 (push)** — [`crate::prompt::Call2Prompt`] writes Gaia's reply,
//!    grounded in that context. Its `response.json.text` is the answer we return.
//!
//! The engine is **infallible by design**: any model or network failure is
//! captured into the reply text with an `error` verdict rather than dropped, so
//! the caller always has something to show the user. When no LLM is configured
//! (the default outside `GAIA_MODE=dev`), it returns a clear skeleton reply so
//! the front end still works end-to-end.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::cosmos::CosmosClient;
use crate::embeddings::EmbeddingClient;
use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt, Call2Prompt};
use crate::pull_data_controller::PullDataController;
use crate::push_data_controller::{PushActionTiming, PushDataController};
use crate::web_search::BraveClient;
use crate::write_data_controller::{WriteDataController, DEFAULT_INDEX_DIMS};

/// The result of one turn, serialized as the front end's `ReplyResult`.
///
/// Field names mirror `web/src/lib/types.ts`: `thoughtId` is camelCase on the
/// wire, and the optional fields are omitted entirely when empty so the JSON
/// stays compact.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TurnResult {
    /// Gaia's reply text to show the user.
    pub reply: String,
    /// Safety verdict for the turn (`allow` on success, `error` on failure).
    pub verdict: String,
    /// Which backend produced the reply (model name, or `skeleton`).
    pub routing: String,
    /// Average attention/emotion score for the turn (0.0 when not computed).
    pub attention: f64,
    /// Stable id for this thought, useful for client-side correlation.
    #[serde(rename = "thoughtId")]
    pub thought_id: String,
    /// Web-search queries run during the turn, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub searches: Option<Vec<String>>,
    /// A short, human-readable summary of the side effects LLM Call 2 planned
    /// this turn (WhatsApp / Push / Edwino actuate / store write-backs). The
    /// front end renders it as an extra "actions performed" bubble. Omitted when
    /// the turn planned no actions (e.g. skeleton mode or a Call 2 failure).
    #[serde(rename = "actionsSummary", skip_serializing_if = "Option::is_none")]
    pub actions_summary: Option<String>,
    /// Diagnostics for the **pull pass** (LLM Call 1 + retrieval), shown in the
    /// UI debug panel. Omitted in skeleton mode (no model configured).
    #[serde(rename = "pullDebug", skip_serializing_if = "Option::is_none")]
    pub pull_debug: Option<PullDebug>,
    /// Diagnostics for the **push pass** (LLM Call 2 + planned side effects),
    /// shown in the UI debug panel. Omitted in skeleton mode and when Call 2
    /// failed before producing a reply.
    #[serde(rename = "pushDebug", skip_serializing_if = "Option::is_none")]
    pub push_debug: Option<PushDebug>,
    /// Persistence status for this turn's mandatory Cosmos write-back. Surfaced
    /// to the user (not just the debug panel) so any failure to connect to
    /// Cosmos or to write is impossible to miss. Present on every live turn that
    /// produced a reply; omitted only in skeleton mode (no model, nothing to
    /// persist) and when Call 2 failed before producing a reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write: Option<WriteStatus>,
}

/// Visible persistence status for the turn's Cosmos write-back.
///
/// Serialized as the front end's `WriteStatus`. Writes are **mandatory**: when
/// the write controller is missing/offline, or a Cosmos read/write fails, this
/// carries `ok = false` and a human-readable `detail` the UI renders as a
/// prominent error banner rather than a silent skip.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WriteStatus {
    /// `true` when the turn was persisted (or was an idempotent replay no-op);
    /// `false` when persistence was skipped (offline) or failed.
    pub ok: bool,
    /// Human-readable detail: a short confirmation on success (id, action,
    /// size), or the underlying error on failure.
    pub detail: String,
}

/// Debug diagnostics for the pull pass (LLM Call 1 + retrieval).
///
/// Serialized as the front end's `PullDebug`: which model answered Call 1, how
/// long that call took, and the retrieval actions the model chose this turn.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PullDebug {
    /// The model that produced LLM Call 1 (e.g. `gpt-4o-mini`).
    pub model: String,
    /// Wall-clock milliseconds spent in the LLM Call 1 request.
    #[serde(rename = "llmMs")]
    pub llm_ms: u64,
    /// The retrieval actions Call 1 planned, e.g. `q1 → Web`, `q3 → GaiaKB`.
    /// Empty when Call 1 produced no parseable action plan.
    pub actions: Vec<String>,
}

/// Debug diagnostics for the push pass (LLM Call 2 + planned side effects).
///
/// Serialized as the front end's `PushDebug`: which model answered Call 2, how
/// long that call took, and the per-action type + processing time of every side
/// effect the model planned.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PushDebug {
    /// The model that produced LLM Call 2 (e.g. `gpt-4o-mini`).
    pub model: String,
    /// Wall-clock milliseconds spent in the LLM Call 2 request.
    #[serde(rename = "llmMs")]
    pub llm_ms: u64,
    /// One entry per planned action: its type and time-to-process (ms).
    pub actions: Vec<PushActionTiming>,
}

/// Runs Gaia turns. Holds the (optional) model and web-search clients plus the
/// default user id, and is cheap to clone/share across connection threads.
#[derive(Debug, Clone)]
pub struct Engine {
    /// The chat model client, or `None` to run in skeleton mode.
    llm: Option<LlmClient>,
    /// The web-search client, or `None` to skip web search.
    web_search: Option<BraveClient>,
    /// The Cosmos client used to execute the plan's data-lake/KB/diary/
    /// connections queries, or `None` when Cosmos is not configured.
    cosmos: Option<CosmosClient>,
    /// The embedding client used when a retrieval action chooses semantic mode,
    /// or `None` when embeddings are not configured.
    embedder: Option<EmbeddingClient>,
    /// The shared write controller used to persist each completed turn back to
    /// Cosmos (append-and-re-embed). Built from the same Cosmos + embedding
    /// clients; `None` (or offline) means the engine simply skips persistence.
    writer: Option<WriteDataController>,
}

impl Engine {
    /// Build an engine from the process environment.
    ///
    /// Reuses the exact same configuration the console app uses:
    /// [`LlmClient::from_env`] (active when `GAIA_MODE=dev`/`local`) and
    /// [`BraveClient::from_env`]. Returns the engine plus any non-fatal warning
    /// (e.g. dev mode requested but the model is misconfigured) so the server can
    /// log it. A configuration error never prevents the server from starting; it
    /// simply falls back to skeleton replies.
    pub fn from_env() -> (Self, Option<String>) {
        // Collect any non-fatal configuration problems; none of them prevent the
        // server from starting (it just degrades the affected capability).
        let mut warnings: Vec<String> = Vec::new();

        let llm = match LlmClient::from_env() {
            Ok(client) => client,
            Err(err) => {
                warnings.push(err.to_string());
                None
            }
        };
        // The retrieval clients are only meaningful alongside a model; mirror the
        // console app and the self-test and only wire them when the LLM is active.
        let (web_search, cosmos, embedder) = if llm.is_some() {
            let web_search = BraveClient::from_env();
            let cosmos = match CosmosClient::from_env() {
                Ok(client) => client,
                Err(err) => {
                    warnings.push(format!("Cosmos retrieval disabled: {err}"));
                    None
                }
            };
            let embedder = match EmbeddingClient::from_env() {
                Ok(client) => client,
                Err(err) => {
                    warnings.push(format!("semantic embeddings disabled: {err}"));
                    None
                }
            };
            (web_search, cosmos, embedder)
        } else {
            (None, None, None)
        };

        let warning = if warnings.is_empty() {
            None
        } else {
            Some(warnings.join("; "))
        };
        // Build the write controller from the *same* Cosmos + embedding clients
        // the rest of the engine uses (both are cheap to clone). When either is
        // absent the controller is offline and persistence is silently skipped,
        // so wiring it in never destabilizes a turn. The index size honours the
        // `DATALAKE_INDEX_VECTOR_DIMS` override, falling back to the default.
        let writer = if cosmos.is_some() && embedder.is_some() {
            let index_dims = crate::llm::value_from_env("DATALAKE_INDEX_VECTOR_DIMS")
                .and_then(|raw| raw.trim().parse::<usize>().ok())
                .filter(|dims| *dims > 0)
                .unwrap_or(DEFAULT_INDEX_DIMS);
            Some(WriteDataController::new(
                cosmos.clone(),
                embedder.clone(),
                index_dims,
            ))
        } else {
            None
        };
        (
            Engine {
                llm,
                web_search,
                cosmos,
                embedder,
                writer,
            },
            warning,
        )
    }

    /// Construct an engine directly from its parts (used by tests).
    ///
    /// Cosmos and embeddings default to `None`; the existing tests exercise the
    /// skeleton and prompt paths, which do not touch live retrieval.
    #[cfg(test)]
    pub fn new(llm: Option<LlmClient>, web_search: Option<BraveClient>) -> Self {
        Engine {
            llm,
            web_search,
            cosmos: None,
            embedder: None,
            writer: None,
        }
    }

    /// A human-readable, secret-free summary of how the engine is configured,
    /// for the server's startup banner.
    pub fn describe(&self) -> String {
        match &self.llm {
            Some(client) => {
                let web = if self.web_search.is_some() {
                    "web search ON"
                } else {
                    "web search off"
                };
                let cosmos = if self.cosmos.is_some() {
                    "Cosmos ON"
                } else {
                    "Cosmos off"
                };
                let embed = if self.embedder.is_some() {
                    "embeddings ON"
                } else {
                    "embeddings off"
                };
                let writes = if self.writer.as_ref().is_some_and(|w| w.is_online()) {
                    "writes ON"
                } else {
                    "writes off"
                };
                format!(
                    "live model {} at {} ({web}, {cosmos}, {embed}, {writes})",
                    client.model(),
                    client.endpoint()
                )
            }
            None => "skeleton mode (set GAIA_MODE=dev to enable live model calls)".to_string(),
        }
    }

    /// Run one full turn for `user_id` and `input`, returning the reply.
    ///
    /// Every read and write this turn is scoped to `user_id` (user isolation).
    /// The call is blocking and always returns a [`TurnResult`]; failures are
    /// folded into the reply rather than surfaced as an error.
    pub fn run_turn(&self, user_id: &str, input: &str) -> TurnResult {
        let thought_id = new_thought_id();
        let input = input.trim();

        // No model configured: return a clear, friendly skeleton reply so the
        // front end is still usable without any Azure/GitHub credentials.
        let Some(client) = &self.llm else {
            return TurnResult {
                reply: skeleton_reply(user_id, input),
                verdict: "allow".to_string(),
                routing: "skeleton".to_string(),
                attention: 0.0,
                thought_id,
                searches: None,
                actions_summary: None,
                pull_debug: None,
                push_debug: None,
                // Skeleton mode persists nothing (no model, nothing to save).
                write: None,
            };
        };

        let requested_at = now_rfc3339();
        let model = client.model().to_string();

        // --- LLM Call 1: the pull / research pass --------------------------
        let call1 = Call1Prompt::build(user_id, input, "", &requested_at);
        let call1_start = Instant::now();
        let call1_raw = match client.complete(&call1.system, &call1.user) {
            Ok(text) => text,
            Err(err) => {
                // Call 1 failed: we can still answer, just without a research plan.
                format!("(LLM Call 1 failed: {err})")
            }
        };
        let call1_ms = call1_start.elapsed().as_millis() as u64;

        // --- Retrieval + deterministic Response Data Context ---------------
        // Execute every retrieval action Call 1 planned and fold the results,
        // together with Call 1's analysis/facts/newContext, into the eight-
        // section markdown that grounds Call 2. This is the exact same builder
        // the data-retrieval self-test uses, so the cloud app and the test stay
        // in lock-step. It never makes an extra LLM call.
        let (response_data_context, searches, pull_actions) =
            self.assemble_context(user_id, input, &requested_at, &call1_raw);

        // Diagnostics for the pull pass shown in the UI debug panel: which model
        // ran Call 1, how long it took, and the retrieval actions it chose.
        let pull_debug = PullDebug {
            model: model.clone(),
            llm_ms: call1_ms,
            actions: pull_actions,
        };

        // --- LLM Call 2: the push / answer pass ----------------------------
        let call2 = Call2Prompt::build(user_id, input, &response_data_context, &requested_at);
        let call2_start = Instant::now();
        let call2_result = client.complete(&call2.system, &call2.user);
        let call2_ms = call2_start.elapsed().as_millis() as u64;
        match call2_result {
            Ok(call2_raw) => {
                // Drive the shared push controller — the exact same parsing and
                // audit the data-execution self-test runs. It yields both the
                // reply text to show and the planned-side-effects summary bubble.
                let push = PushDataController::process(&call2_raw);
                // Capture the actions summary before moving the reply text into
                // the result struct (both borrow `push`).
                let actions_summary = push.actions_summary();
                // Per-action type + processing time for the UI debug panel.
                let push_action_timings = push
                    .actions
                    .as_ref()
                    .map(crate::push_data_controller::time_actions)
                    .unwrap_or_default();
                // Persist this completed exchange to the user's personal data
                // lake (append-and-re-embed). Writes are mandatory: the returned
                // status is surfaced to the user so a Cosmos connection or write
                // failure is visible rather than silently dropped.
                let write = self.persist_turn(user_id, input, &push.reply_text, &requested_at);
                TurnResult {
                    reply: push.reply_text,
                    verdict: "allow".to_string(),
                    routing: model.clone(),
                    attention: 0.0,
                    thought_id,
                    searches: if searches.is_empty() {
                        None
                    } else {
                        Some(searches)
                    },
                    // Audit Call 2's actions.json and surface a short summary bubble.
                    actions_summary,
                    pull_debug: Some(pull_debug),
                    push_debug: Some(PushDebug {
                        model,
                        llm_ms: call2_ms,
                        actions: push_action_timings,
                    }),
                    write: Some(write),
                }
            }
            Err(err) => TurnResult {
                reply: format!("Sorry, I could not complete my reply: {err}"),
                verdict: "error".to_string(),
                routing: model.clone(),
                attention: 0.0,
                thought_id,
                searches: if searches.is_empty() {
                    None
                } else {
                    Some(searches)
                },
                actions_summary: None,
                pull_debug: Some(pull_debug),
                // Call 2 failed: report the model and call time, but no actions.
                push_debug: Some(PushDebug {
                    model,
                    llm_ms: call2_ms,
                    actions: Vec::new(),
                }),
                // No reply was produced, so there is nothing to persist this turn.
                write: None,
            },
        }
    }

    /// Execute this turn's retrieval plan and deterministically assemble the
    /// Response Data Context handed to LLM Call 2.
    ///
    /// Mirrors the data-retrieval self-test: parse Call 1's `actions.json` plus
    /// its analysis/facts/newContext documents, run the Cosmos-backed queries
    /// (via [`Executor`]) and the `Web` searches (via Brave), then fold every
    /// result into the eight-section markdown via
    /// [`build_response_data_context`]. It makes **no** extra LLM call and never
    /// fails: a missing client or an unparsable plan simply yields empty
    /// sections, so Call 2 always receives a stable, predictable structure.
    ///
    /// Returns the context markdown, the web queries that actually ran (for the
    /// [`TurnResult::searches`] field), and a short label per retrieval action
    /// the plan chose (for the pull-pass debug panel, e.g. `q1 → Web`).
    fn assemble_context(
        &self,
        user_id: &str,
        input: &str,
        requested_at: &str,
        call1_raw: &str,
    ) -> (String, Vec<String>, Vec<String>) {
        // Drive the shared pull controller — the exact same code the
        // data-retrieval self-test runs — and keep only what the reply needs:
        // the assembled context and the web queries that ran. We pass
        // `capture_queries = false` because the cloud app does not need the
        // per-query SQL audit (and capturing it would cost an extra embedding
        // call per semantic action).
        let result = PullDataController::new(
            self.cosmos.as_ref(),
            self.embedder.as_ref(),
            self.web_search.as_ref(),
        )
        .execute(user_id, input, requested_at, call1_raw, false);

        // One readable label per planned retrieval action (`id → target`), in
        // plan order, so the debug panel shows exactly what Call 1 chose to fetch.
        let actions = result
            .plan
            .as_ref()
            .map(|plan| {
                plan.actions
                    .iter()
                    .map(|a| format!("{} → {}", a.id, a.target))
                    .collect()
            })
            .unwrap_or_default();

        (result.context, result.searches, actions)
    }

    /// Persist a completed exchange to the user's personal data lake, returning
    /// a [`WriteStatus`] the caller surfaces to the user.
    ///
    /// Appends a single `User: … / Gaia: …` chunk to today's `UsersDataLake`
    /// record for `user_id` through the shared [`WriteDataController`], which
    /// reads the day's record, appends, re-embeds the whole day once, and writes
    /// it back (creating it on the first turn of the day). `user_id` is the
    /// `/userId` partition, so every write stays scoped to its owner.
    ///
    /// Writes are **mandatory**: unlike the rest of the engine's best-effort
    /// degradation, a missing/offline writer or a Cosmos read/write failure is
    /// reported back as `WriteStatus { ok: false, .. }` (and logged to stderr)
    /// so the front end can show a visible error instead of silently losing the
    /// turn. The reply text itself is never altered by a write failure.
    fn persist_turn(
        &self,
        user_id: &str,
        input: &str,
        reply: &str,
        now_rfc3339: &str,
    ) -> WriteStatus {
        // A missing or offline writer is a hard, visible error — not a silent
        // skip — because persistence is required for every live turn.
        let Some(writer) = self.writer.as_ref() else {
            let detail = "Cosmos write-back is not configured (no Cosmos and/or embedding client)"
                .to_string();
            eprintln!("persist turn skipped for user {user_id}: {detail}");
            return WriteStatus { ok: false, detail };
        };
        if !writer.is_online() {
            let detail =
                "Cosmos write-back is offline (Cosmos and/or embedding client not connected)"
                    .to_string();
            eprintln!("persist turn skipped for user {user_id}: {detail}");
            return WriteStatus { ok: false, detail };
        }

        // One readable line capturing both sides of the exchange. The controller
        // timestamps and appends it under today's record.
        let chunk = format!("User: {input}\nGaia: {reply}");
        match writer.upsert_daily("UsersDataLake", user_id, now_rfc3339, &chunk) {
            Ok(outcome) => {
                // A lightweight, secret-free trace so operators can see writes
                // landing without inspecting Cosmos directly.
                eprintln!(
                    "persisted turn: {} ({}, {} bytes, vector {}d)",
                    outcome.id,
                    outcome.action.label(),
                    outcome.data_bytes,
                    outcome.vector_dims,
                );
                WriteStatus {
                    ok: true,
                    detail: format!(
                        "saved to Cosmos UsersDataLake: {} ({}, {} bytes)",
                        outcome.id,
                        outcome.action.label(),
                        outcome.data_bytes,
                    ),
                }
            }
            Err(err) => {
                let detail = format!("Cosmos write failed: {err}");
                eprintln!("persist turn failed for user {user_id}: {detail}");
                WriteStatus { ok: false, detail }
            }
        }
    }
}

/// Build the skeleton-mode reply shown when no model is configured.
fn skeleton_reply(user_id: &str, input: &str) -> String {
    let who = if user_id.is_empty() { "there" } else { user_id };
    if input.is_empty() {
        format!("Hello {who}. Gaia's backend is running in skeleton mode — set GAIA_MODE=dev (and a model) to enable live replies.")
    } else {
        format!(
            "Hello {who}. I received: \"{input}\". Gaia's backend is running in skeleton mode \
             (no model configured), so this is a stub reply. Set GAIA_MODE=dev to enable the \
             real thought sequence."
        )
    }
}

/// Process-wide counter making generated ids unique even within the same
/// nanosecond, so concurrent turns never collide on a thought id.
static THOUGHT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique, non-cryptographic thought id like `th_18f2c…`.
///
/// This is an identifier, not a secret: it combines the current time with a
/// monotonic counter, which is enough to correlate a reply on the client.
fn new_thought_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = THOUGHT_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("th_{nanos:x}{seq:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_engine_echoes_input() {
        let engine = Engine::new(None, None);
        let result = engine.run_turn("alice", "hello");
        assert_eq!(result.routing, "skeleton");
        assert_eq!(result.verdict, "allow");
        assert!(result.reply.contains("hello"));
        assert!(result.reply.contains("alice"));
        assert!(result.thought_id.starts_with("th_"));
    }

    #[test]
    fn thought_ids_are_unique() {
        let a = new_thought_id();
        let b = new_thought_id();
        assert_ne!(a, b);
    }

    #[test]
    fn persist_turn_reports_a_visible_error_without_an_online_writer() {
        // Writes are mandatory: the skeleton engine has no writer configured, so
        // persisting a turn must not silently skip. It returns a visible failure
        // status (never panics) the front end can show to the user.
        let engine = Engine::new(None, None);
        let status = engine.persist_turn("alice", "hello", "hi alice", "2026-06-27T10:00:00Z");
        assert!(!status.ok, "missing writer must report a failed write");
        assert!(
            status.detail.to_lowercase().contains("cosmos"),
            "detail should name Cosmos, got: {}",
            status.detail
        );
    }

    #[test]
    fn assemble_context_is_deterministic_and_degrades_without_clients() {
        // No web/Cosmos/embedder clients configured: the retrieval sections must
        // still be emitted (empty), Call 1's analysis/facts/newContext folded in,
        // and no web search recorded — and the output must be byte-stable.
        let engine = Engine::new(None, None);
        let reply = r#"[
          [ { "id": "q1", "kind": "search", "target": "Web", "intent": "mars news" } ],
          { "emotion": "calm", "truthfulness": "honest", "intention": "learn" },
          [ { "fact": "favourite_colour", "value": "blue" } ],
          { "summary": "We spoke about colours before." }
        ]"#;

        let (context, searches, actions) =
            engine.assemble_context("alice", "hi", "2026-06-21T00:00:00Z", reply);

        for heading in [
            "## WebSearchResults",
            "## DataLakeResults",
            "## KnowledgeBaseResults",
            "## ConnectionsResults",
            "## EmotionResults",
            "## TruthfulNessResults",
            "## IntentionResults",
            "## OldContextSummary",
        ] {
            assert!(context.contains(heading), "missing heading {heading}");
        }
        // Call 1's extras are folded into their sections.
        assert!(context.contains("calm"));
        assert!(context.contains("We spoke about colours before."));
        assert!(context.contains("**favourite_colour:** blue"));
        // With no Brave client, no web search ran this turn.
        assert!(searches.is_empty());

        // Determinism: identical inputs produce identical output.
        let (again, _, actions_again) =
            engine.assemble_context("alice", "hi", "2026-06-21T00:00:00Z", reply);
        assert_eq!(context, again);
        // The pull-action labels are also stable across identical inputs.
        assert_eq!(actions, actions_again);
    }

    /// Wrap `content` in the chat-completions envelope the LLM client parses.
    fn completion(content: &str) -> String {
        let escaped = content.replace('\\', "\\\\").replace('"', "\\\"");
        format!(r#"{{"choices":[{{"message":{{"content":"{escaped}"}}}}]}}"#)
    }

    #[test]
    fn run_turn_drives_two_llm_calls_and_returns_the_pushed_reply() {
        // With a live model but no retrieval/write clients, a full turn issues
        // exactly two completions (the pull pass and the push pass). The mock
        // answers them in order; Call 2's content becomes the visible reply.
        let (endpoint, handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), completion("call-1 analysis")),
            (
                "200 OK".to_string(),
                completion("Hello Alice, here is your answer."),
            ),
        ]);
        let llm = LlmClient::for_test(endpoint);
        let engine = Engine::new(Some(llm), None);

        let result = engine.run_turn("alice", "what's up?");

        assert_eq!(result.reply, "Hello Alice, here is your answer.");
        assert_eq!(result.verdict, "allow");
        assert_eq!(result.routing, "gpt-test");
        // Both debug panels are populated when both calls succeed.
        assert!(result.pull_debug.is_some());
        assert!(result.push_debug.is_some());
        // Persistence is mandatory and always reported. This test engine has no
        // writer, so the status must be present and flag the missing Cosmos.
        let write = result
            .write
            .expect("a completed turn always reports a write status");
        assert!(
            !write.ok,
            "no writer configured, so the write must be reported as failed"
        );
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn run_turn_reports_a_failed_second_call_as_an_error_verdict() {
        // Call 1 succeeds, Call 2 returns a 500: the turn degrades to an error
        // verdict with an apologetic reply rather than panicking.
        let (endpoint, handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), completion("call-1 analysis")),
            (
                "500 Internal Server Error".to_string(),
                r#"{"error":"boom"}"#.to_string(),
            ),
        ]);
        let llm = LlmClient::for_test(endpoint);
        let engine = Engine::new(Some(llm), None);

        let result = engine.run_turn("alice", "hi");

        assert_eq!(result.verdict, "error");
        assert!(result.reply.contains("could not complete"));
        // The push debug panel still reports the model, just with no actions.
        assert!(result.push_debug.is_some());
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn describe_reports_a_live_model_and_its_capabilities() {
        // A configured model with no other clients should describe itself as
        // live with web search, Cosmos, embeddings, and writes all off.
        // `describe` only reads the client's model/endpoint, so no server is
        // needed — any endpoint string will do.
        let engine = Engine::new(
            Some(LlmClient::for_test("http://127.0.0.1:9/".to_string())),
            None,
        );
        let summary = engine.describe();
        assert!(summary.starts_with("live model gpt-test"));
        assert!(summary.contains("web search off"));
        assert!(summary.contains("Cosmos off"));
        assert!(summary.contains("writes off"));
    }

    #[test]
    fn describe_reports_skeleton_mode_without_a_model() {
        let engine = Engine::new(None, None);
        assert!(engine.describe().contains("skeleton mode"));
    }
}
