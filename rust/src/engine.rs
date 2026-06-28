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
use crate::health::{DependencyHealth, HealthReport};
use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt, Call2Prompt};
use crate::pull_data_controller::{PullActionTiming, PullDataController};
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
    /// The live process-log for this turn: one ordered [`TurnEvent`] per phase
    /// (pull model, retrieval, push model, persistence) plus any warning or
    /// error encountered. These are streamed to the client as they happen (over
    /// the WebSocket) and also returned in full so the debug panel can replay
    /// them after the fact. Omitted from the JSON when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<TurnEvent>,
}

/// One streamed entry in a turn's live process-log.
///
/// Serialized as the front end's `TurnEvent`. The engine emits these as it moves
/// through a turn so the user can see *what Gaia is doing right now* — running
/// the pull model, assembling context, running the push model, persisting — and
/// any warning or error is surfaced immediately rather than hidden in stderr.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TurnEvent {
    /// Monotonic 0-based sequence number within the turn, for stable ordering.
    pub seq: u64,
    /// Which phase produced the event, e.g. `turn`, `pull`, `retrieval`,
    /// `push`, or `persist`.
    pub phase: String,
    /// Severity: `info` for normal progress, `warn` for a non-fatal degradation,
    /// or `error` for a failure that changed the outcome.
    pub level: String,
    /// Human-readable description of what just happened.
    pub message: String,
    /// Wall-clock milliseconds for the operation this event closes, when the
    /// event marks the end of a timed step (e.g. the pull model call). Omitted
    /// for plain progress markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<f64>,
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
    /// Per-operation write latency — one [`WriteTiming`] for every Cosmos write
    /// this turn (the `GaiaDataLake` exchange record, each planned store
    /// upsert, and each connection-ledger delta), in execution order, with the
    /// actual wall-clock milliseconds it took. Lets the UI show the real latency
    /// of every write rather than a single aggregate. Omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub operations: Vec<WriteTiming>,
}

/// The actual wall-clock latency of one Cosmos write, surfaced to the UI.
///
/// Serialized as the front end's `WriteTiming`. Unlike the push-pass
/// [`PushActionTiming`] (which times the in-memory *audit* of a planned action),
/// this measures the real network round-trip of the write itself.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WriteTiming {
    /// A short label for the write, e.g. `GaiaDataLake`, `Upsert GaiaDiary`,
    /// or `GaiaConnections delta`.
    #[serde(rename = "type")]
    pub label: String,
    /// Wall-clock milliseconds the write took (may be fractional).
    pub ms: f64,
    /// `true` when the write succeeded, `false` when it failed.
    pub ok: bool,
}

/// Collects and (optionally) streams a turn's [`TurnEvent`] process-log.
///
/// The engine threads one reporter through a turn. Every `info`/`warn`/`error`
/// call assigns the next sequence number, hands the event to the streaming
/// `sink` (so the WebSocket route can forward it to the client immediately), and
/// keeps a copy so the finished [`TurnResult`] carries the whole log for replay.
/// Non-streaming callers pass a no-op sink and simply receive the collected log.
struct TurnReporter<'a> {
    /// The next sequence number to assign.
    seq: u64,
    /// Every event emitted so far, in order.
    events: Vec<TurnEvent>,
    /// Live sink invoked with each event as it is emitted.
    sink: &'a mut dyn FnMut(&TurnEvent),
}

impl<'a> TurnReporter<'a> {
    /// Create a reporter that streams to `sink` and collects every event.
    fn new(sink: &'a mut dyn FnMut(&TurnEvent)) -> Self {
        TurnReporter {
            seq: 0,
            events: Vec::new(),
            sink,
        }
    }

    /// Emit one event: assign its sequence number, stream it, and keep it.
    fn emit(&mut self, phase: &str, level: &str, message: String, ms: Option<f64>) {
        let event = TurnEvent {
            seq: self.seq,
            phase: phase.to_string(),
            level: level.to_string(),
            message,
            ms,
        };
        self.seq += 1;
        // Hand the event to the live stream first so the client sees it with the
        // least delay, then retain a copy for the final result.
        (self.sink)(&event);
        self.events.push(event);
    }

    /// Emit a normal progress marker.
    fn info(&mut self, phase: &str, message: String) {
        self.emit(phase, "info", message, None);
    }

    /// Emit a progress marker that closes a timed step (carries its `ms`).
    fn timed(&mut self, phase: &str, message: String, ms: u64) {
        self.emit(phase, "info", message, Some(ms as f64));
    }

    /// Emit a progress marker that closes a timed step, carrying a fractional
    /// `ms` (used for sub-millisecond retrieval reads whose precise latency
    /// would be lost by rounding to whole milliseconds).
    fn timed_f64(&mut self, phase: &str, message: String, ms: f64) {
        self.emit(phase, "info", message, Some(ms));
    }

    /// Emit a non-fatal warning that closes a timed step.
    fn warn_timed(&mut self, phase: &str, message: String, ms: u64) {
        self.emit(phase, "warn", message, Some(ms as f64));
    }

    /// Emit a failure that closes a timed step.
    fn error_timed(&mut self, phase: &str, message: String, ms: u64) {
        self.emit(phase, "error", message, Some(ms as f64));
    }

    /// Emit a warning with no associated duration.
    fn warn(&mut self, phase: &str, message: String) {
        self.emit(phase, "warn", message, None);
    }

    /// Record one finished Cosmos write as a `persist`-phase event, choosing the
    /// severity from whether it succeeded and carrying its real latency.
    fn persist_write(&mut self, status: &WriteStatus, ms: f64) {
        let level = if status.ok { "info" } else { "error" };
        self.emit("persist", level, status.detail.clone(), Some(ms));
    }

    /// Drain and return every collected event (used once, when the turn ends).
    fn take_events(&mut self) -> Vec<TurnEvent> {
        std::mem::take(&mut self.events)
    }
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
    /// The retrieval actions Call 1 planned, each with its label (e.g.
    /// `q3 → GaiaKB`) and the wall-clock milliseconds the read took. Empty when
    /// Call 1 produced no parseable action plan.
    pub actions: Vec<PullActionTiming>,
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

    /// Actively probe every configured dependency and return a readiness report.
    ///
    /// This backs the server's `/readyz` endpoint. For each external dependency
    /// the engine is wired to — the Foundry model-router, the Foundry embedding
    /// deployment, Cosmos DB, and Brave Search — it runs a real minimal request
    /// that exercises connectivity *and* the required RBAC. Dependencies that
    /// are not configured in this deployment are reported as `skipped` and never
    /// make the service unready. The call is blocking (it performs live network
    /// requests) and is intended for an explicit readiness probe, not hot paths.
    pub fn check_health(&self) -> HealthReport {
        let mut checks = Vec::with_capacity(4);

        // Foundry model-router (LLM): a one-token completion validates the
        // endpoint, the managed-identity/API-key auth, and the OpenAI User role.
        checks.push(match &self.llm {
            Some(client) => match client.ping() {
                Ok(()) => DependencyHealth::ok(
                    "foundry-model-router",
                    format!("model {} at {}", client.model(), client.endpoint()),
                ),
                Err(e) => DependencyHealth::failed("foundry-model-router", e.to_string()),
            },
            None => DependencyHealth::skipped("foundry-model-router"),
        });

        // Foundry embeddings: shares the Foundry account/RBAC but uses a
        // different deployment, so probe it independently.
        checks.push(match &self.embedder {
            Some(client) => match client.ping() {
                Ok(()) => DependencyHealth::ok("foundry-embeddings", client.endpoint().to_string()),
                Err(e) => DependencyHealth::failed("foundry-embeddings", e.to_string()),
            },
            None => DependencyHealth::skipped("foundry-embeddings"),
        });

        // Cosmos DB: an authenticated metadata read of the target database.
        checks.push(match &self.cosmos {
            Some(client) => match client.ping() {
                Ok(()) => DependencyHealth::ok(
                    "cosmos",
                    format!("{} db={}", client.endpoint(), client.database()),
                ),
                Err(e) => DependencyHealth::failed("cosmos", e.to_string()),
            },
            None => DependencyHealth::skipped("cosmos"),
        });

        // Brave Search: a single one-result query validates the subscription key.
        checks.push(match &self.web_search {
            Some(client) => match client.ping() {
                Ok(()) => DependencyHealth::ok("brave-search", client.endpoint().to_string()),
                Err(e) => DependencyHealth::failed("brave-search", e.to_string()),
            },
            None => DependencyHealth::skipped("brave-search"),
        });

        HealthReport::from_checks(checks)
    }

    /// Run one full turn for `user_id` and `input`, returning the reply.
    ///
    /// Every read and write this turn is scoped to `user_id` (user isolation).
    /// The call is blocking and always returns a [`TurnResult`]; failures are
    /// folded into the reply rather than surfaced as an error.
    pub fn run_turn(&self, user_id: &str, input: &str) -> TurnResult {
        // Non-streaming callers (the POST route and tests) don't observe the
        // process-log live; they still receive every event on the result via
        // [`TurnResult::events`]. A no-op sink keeps the same code path.
        self.run_turn_reported(user_id, input, &mut |_event| {})
    }

    /// Like [`Engine::run_turn`], but reports each phase to `sink` as a
    /// [`TurnEvent`] the moment it happens.
    ///
    /// The WebSocket route passes a sink that forwards every event to the client
    /// as a frame, so the user sees a live process-log ("running pull model…",
    /// "context assembled", warnings, errors). Every event is also collected on
    /// the returned [`TurnResult::events`] for later replay in the debug panel.
    pub fn run_turn_reported(
        &self,
        user_id: &str,
        input: &str,
        sink: &mut dyn FnMut(&TurnEvent),
    ) -> TurnResult {
        let thought_id = new_thought_id();
        let input = input.trim();
        // One reporter threads the whole turn: it streams each event to `sink`
        // and keeps a copy for the finished result.
        let mut reporter = TurnReporter::new(sink);

        // No model configured: return a clear, friendly skeleton reply so the
        // front end is still usable without any Azure/GitHub credentials.
        let Some(client) = &self.llm else {
            reporter.info("turn", "Skeleton mode: no model configured.".to_string());
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
                events: reporter.take_events(),
            };
        };

        let requested_at = now_rfc3339();
        // A concise turn-start marker: WebSocket turns produce no access-log line
        // (the upgrade returns before the logger), so this is the only signal
        // that a turn was entered. It also bookends the per-phase timings below,
        // making a hang easy to localise to Call 1, retrieval, Call 2, or write.
        eprintln!(
            "turn start: user={user_id} thought={thought_id} ({} input chars)",
            input.len()
        );
        reporter.info("turn", "Turn started.".to_string());
        // The configured deployment name (e.g. `model-router`). Used as the
        // routing label and as a fallback when a response doesn't report which
        // underlying model actually ran.
        let model = client.model().to_string();

        // --- LLM Call 1: the pull / research pass --------------------------
        let call1 = Call1Prompt::build(user_id, input, "", &requested_at);
        reporter.info("pull", "Running pull model (LLM Call 1)…".to_string());
        let call1_start = Instant::now();
        // Capture both the text and the model the backend reported. For the
        // model-router, `call1_model` is the underlying model it selected.
        let call1_outcome = client.complete(&call1.system, &call1.user);
        let call1_ms = call1_start.elapsed().as_millis() as u64;
        eprintln!("turn {thought_id}: Call 1 done in {call1_ms}ms");
        let (call1_raw, call1_model) = match call1_outcome {
            Ok(completion) => {
                reporter.timed("pull", "Pull model responded.".to_string(), call1_ms);
                (completion.content, completion.model)
            }
            Err(err) => {
                // Call 1 failed: we can still answer, just without a research plan.
                reporter.warn_timed("pull", format!("LLM Call 1 failed: {err}"), call1_ms);
                (format!("(LLM Call 1 failed: {err})"), None)
            }
        };

        // --- Retrieval + deterministic Response Data Context ---------------
        // Execute every retrieval action Call 1 planned and fold the results,
        // together with Call 1's analysis/facts/newContext, into the eight-
        // section markdown that grounds Call 2. This is the exact same builder
        // the data-retrieval self-test uses, so the cloud app and the test stay
        // in lock-step. It never makes an extra LLM call.
        reporter.info(
            "retrieval",
            "Assembling context (running retrieval)…".to_string(),
        );
        let (response_data_context, searches, pull_actions) =
            self.assemble_context(user_id, input, &requested_at, &call1_raw);
        eprintln!(
            "turn {thought_id}: context assembled ({} web searches)",
            searches.len()
        );
        // Break the single "running retrieval" step down into one timed
        // process-log line per retrieval that actually ran — every Cosmos read
        // (e.g. `q3 → GaiaKB`) and every web search (`q1 → Web`) — with the
        // real wall-clock milliseconds it cost, so the debug panel shows where
        // the retrieval time went instead of a single opaque line.
        for action in &pull_actions {
            reporter.timed_f64(
                "retrieval",
                format!("Retrieved {}", action.action_type),
                action.ms,
            );
        }
        reporter.info(
            "retrieval",
            format!("Context assembled ({} web searches).", searches.len()),
        );

        // Diagnostics for the pull pass shown in the UI debug panel: which model
        // ran Call 1, how long it took, and the retrieval actions it chose. Use
        // the underlying model the router actually selected, falling back to the
        // deployment name when the response didn't report one.
        let pull_debug = PullDebug {
            model: call1_model.clone().unwrap_or_else(|| model.clone()),
            llm_ms: call1_ms,
            actions: pull_actions,
        };

        // --- LLM Call 2: the push / answer pass ----------------------------
        let call2 = Call2Prompt::build(user_id, input, &response_data_context, &requested_at);
        reporter.info("push", "Running push model (LLM Call 2)…".to_string());
        let call2_start = Instant::now();
        let call2_result = client.complete(&call2.system, &call2.user);
        let call2_ms = call2_start.elapsed().as_millis() as u64;
        eprintln!("turn {thought_id}: Call 2 done in {call2_ms}ms");
        let mut result = match call2_result {
            Ok(call2) => {
                reporter.timed("push", "Push model responded.".to_string(), call2_ms);
                // The underlying model the router selected for Call 2 (falls back
                // to the deployment name when the response didn't report one).
                let call2_model = call2.model.clone().unwrap_or_else(|| model.clone());
                let call2_raw = call2.content;
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
                // Persist this completed exchange to Gaia's shared data
                // lake (merge-and-re-embed). Writes are mandatory: the returned
                // status is surfaced to the user so a Cosmos connection or write
                // failure is visible rather than silently dropped. Each write is
                // timed and reported live through the same reporter.
                let write = self.persist_turn(
                    &mut reporter,
                    user_id,
                    input,
                    &push.reply_text,
                    &requested_at,
                    &push.audit,
                );
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
                        model: call2_model,
                        llm_ms: call2_ms,
                        actions: push_action_timings,
                    }),
                    write: Some(write),
                    // Filled in after the match from the reporter's full log.
                    events: Vec::new(),
                }
            }
            Err(err) => {
                reporter.error_timed("push", format!("LLM Call 2 failed: {err}"), call2_ms);
                TurnResult {
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
                    // Filled in after the match from the reporter's full log.
                    events: Vec::new(),
                }
            }
        };

        // Bookend the turn and hand the complete process-log to the result so
        // non-streaming callers (and the debug panel on replay) see every phase.
        reporter.info("turn", "Turn complete.".to_string());
        result.events = reporter.take_events();
        result
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
    /// [`TurnResult::searches`] field), and a per-action timing for every
    /// retrieval the plan executed (for the pull-pass debug panel, e.g.
    /// `q1 → Web` with the milliseconds it cost).
    fn assemble_context(
        &self,
        user_id: &str,
        input: &str,
        requested_at: &str,
        call1_raw: &str,
    ) -> (String, Vec<String>, Vec<PullActionTiming>) {
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

        // One timed entry per planned retrieval action (`id → target` plus the
        // milliseconds it cost), in plan order, so the debug panel shows exactly
        // what Call 1 chose to fetch and how long each read took.
        let actions = result.action_timings;

        (result.context, result.searches, actions)
    }

    /// Persist a completed exchange to Gaia's shared data lake, returning a
    /// [`WriteStatus`] the caller surfaces to the user.
    ///
    /// Merges a single `User: … / Gaia: …` chunk into today's `GaiaDataLake`
    /// record for `user_id` through the shared [`WriteDataController`], which
    /// reads the day's JSON daily log, merges the turn (deduping by content so a
    /// replay never duplicates), re-embeds the day's transcript once, and writes
    /// it back (creating it on the first turn of the day). `user_id` is the
    /// `/entity` partition, so every write stays scoped to its owner.
    ///
    /// Writes are **mandatory**: unlike the rest of the engine's best-effort
    /// degradation, a missing/offline writer or a Cosmos read/write failure is
    /// reported back as `WriteStatus { ok: false, .. }` (and logged to stderr)
    /// so the front end can show a visible error instead of silently losing the
    /// turn. The reply text itself is never altered by a write failure.
    fn persist_turn(
        &self,
        reporter: &mut TurnReporter,
        user_id: &str,
        input: &str,
        reply: &str,
        now_rfc3339: &str,
        audit: &crate::push_data_controller::ActionAudit,
    ) -> WriteStatus {
        // A missing or offline writer is a hard, visible error — not a silent
        // skip — because persistence is required for every live turn.
        let Some(writer) = self.writer.as_ref() else {
            let detail = "Cosmos write-back is not configured (no Cosmos and/or embedding client)"
                .to_string();
            eprintln!("persist turn skipped for user {user_id}: {detail}");
            reporter.warn("persist", detail.clone());
            return WriteStatus {
                ok: false,
                detail,
                operations: Vec::new(),
            };
        };
        if !writer.is_online() {
            let detail =
                "Cosmos write-back is offline (Cosmos and/or embedding client not connected)"
                    .to_string();
            eprintln!("persist turn skipped for user {user_id}: {detail}");
            reporter.warn("persist", detail.clone());
            return WriteStatus {
                ok: false,
                detail,
                operations: Vec::new(),
            };
        }

        reporter.info("persist", "Persisting turn to Cosmos…".to_string());

        // One readable line capturing both sides of the exchange. The controller
        // timestamps and merges it into today's record.
        let chunk = format!("User: {input}\nGaia: {reply}");
        let users_start = Instant::now();
        let users_result = writer.upsert_daily("GaiaDataLake", user_id, now_rfc3339, &chunk);
        let users_ms = users_start.elapsed().as_secs_f64() * 1000.0;
        let users_status = match users_result {
            Ok(outcome) => {
                // A lightweight, secret-free trace so operators can see writes
                // landing without inspecting Cosmos directly.
                eprintln!(
                    "persisted turn: {} ({}, {} bytes, vector {}d, {:.1} ms)",
                    outcome.id,
                    outcome.action.label(),
                    outcome.data_bytes,
                    outcome.vector_dims,
                    users_ms,
                );
                WriteStatus {
                    ok: true,
                    detail: format!(
                        "saved to Cosmos GaiaDataLake: {} ({}, {} bytes, {:.1} ms)",
                        outcome.id,
                        outcome.action.label(),
                        outcome.data_bytes,
                        users_ms,
                    ),
                    operations: Vec::new(),
                }
            }
            Err(err) => {
                let detail = format!("Cosmos write failed: {err}");
                eprintln!("persist turn failed for user {user_id}: {detail}");
                WriteStatus {
                    ok: false,
                    detail,
                    operations: Vec::new(),
                }
            }
        };
        // Report the always-on data-lake write with its real latency so the
        // live process-log shows it the moment it lands (or fails).
        reporter.persist_write(&users_status, users_ms);

        // The shared data lake is the always-on record of the exchange. If it
        // failed (typically a connection problem), the shared-store upserts would
        // fail the same way, so report that one error rather than piling on — but
        // still carry its measured latency so the UI shows the failed write.
        if !users_status.ok {
            return WriteStatus {
                ok: false,
                detail: users_status.detail,
                operations: vec![WriteTiming {
                    label: "GaiaDataLake".to_string(),
                    ms: users_ms,
                    ok: false,
                }],
            };
        }

        // Now execute the *shared* knowledge/data-lake/diary upserts LLM Call 2
        // planned this turn. Previously these were only audited and timed for the
        // debug panel and never written, so GaiaKB/GaiaDataLake/GaiaDiary silently
        // stopped receiving updates. Run them through the same append-and-re-embed
        // path, scoped to the authenticated `user_id` (their `/entity` partition,
        // matching how they are read back). Each `(status, timing)` pair carries
        // both the UI detail line and the real write latency.
        let mut ops: Vec<(WriteStatus, WriteTiming)> = vec![(
            users_status,
            WriteTiming {
                label: "GaiaDataLake".to_string(),
                ms: users_ms,
                ok: true,
            },
        )];
        ops.extend(self.persist_planned_stores(reporter, writer, user_id, now_rfc3339, audit));
        ops.extend(self.persist_planned_connections(reporter, writer, user_id, now_rfc3339, audit));

        // Persistence is mandatory: the turn's status is `ok` only when *every*
        // planned write landed. The detail concatenates each store's line, and the
        // operations vector carries the real per-write latency for the UI panel.
        let ok = ops.iter().all(|(status, _)| status.ok);
        let detail = ops
            .iter()
            .map(|(status, _)| status.detail.clone())
            .collect::<Vec<_>>()
            .join("; ");
        let operations = ops.into_iter().map(|(_, timing)| timing).collect();
        WriteStatus {
            ok,
            detail,
            operations,
        }
    }

    /// Execute the daily-store upserts LLM Call 2 planned this turn and return
    /// one `(status, timing)` pair per store, in plan order.
    ///
    /// Covers the shared `GaiaKB`, `GaiaDataLake`, and `GaiaDiary` stores (the
    /// friendship ledger `GaiaConnections` is handled separately and excluded by
    /// [`crate::push_data_controller::planned_store_writes`]). `GaiaKB` is a
    /// knowledge base, so each fact is saved as its own record via
    /// [`WriteDataController::insert_fact`] (per-fact embedding and salience);
    /// `GaiaDataLake` and `GaiaDiary` stay per-day appends via
    /// [`WriteDataController::upsert_daily`]. Each write is scoped to the
    /// authenticated `user_id`, which is the `/entity` partition key these shared
    /// stores are both written and read under. Every write is timed, reported
    /// live through `reporter`, and a per-store failure is reported (and logged)
    /// but never aborts the others.
    fn persist_planned_stores(
        &self,
        reporter: &mut TurnReporter,
        writer: &WriteDataController,
        user_id: &str,
        now_rfc3339: &str,
        audit: &crate::push_data_controller::ActionAudit,
    ) -> Vec<(WriteStatus, WriteTiming)> {
        crate::push_data_controller::planned_store_writes(audit)
            .into_iter()
            .map(|(store, data, salience)| {
                let start = Instant::now();
                // GaiaKB is a knowledge base: each fact gets its own record so
                // its embedding and salience describe that one fact. The other
                // writer stores (GaiaDataLake, GaiaDiary) stay per-day appends.
                let result = if store == "GaiaKB" {
                    writer.insert_fact(store, user_id, now_rfc3339, &data, salience)
                } else {
                    writer.upsert_daily(store, user_id, now_rfc3339, &data)
                };
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let status = match result {
                    Ok(outcome) => {
                        eprintln!(
                            "persisted store write: {} ({}, {} bytes, vector {}d, {:.1} ms)",
                            outcome.id,
                            outcome.action.label(),
                            outcome.data_bytes,
                            outcome.vector_dims,
                            ms,
                        );
                        WriteStatus {
                            ok: true,
                            detail: format!(
                                "saved to Cosmos {}: {} ({}, {} bytes, {:.1} ms)",
                                store,
                                outcome.id,
                                outcome.action.label(),
                                outcome.data_bytes,
                                ms,
                            ),
                            operations: Vec::new(),
                        }
                    }
                    Err(err) => {
                        let detail = format!("Cosmos write to {store} failed: {err}");
                        eprintln!("store write failed for user {user_id}: {detail}");
                        WriteStatus {
                            ok: false,
                            detail,
                            operations: Vec::new(),
                        }
                    }
                };
                reporter.persist_write(&status, ms);
                let timing = WriteTiming {
                    label: format!("Upsert {store}"),
                    ms,
                    ok: status.ok,
                };
                (status, timing)
            })
            .collect()
    }

    /// Append the friendship-ledger deltas LLM Call 2 planned this turn to
    /// `GaiaConnections`, returning one `(status, timing)` pair per delta, in
    /// plan order.
    ///
    /// The ledger is append-only and balance-carrying, so this uses the
    /// dedicated [`WriteDataController::append_connection_delta`] path rather
    /// than `upsert_daily`. Like the snapshot stores it is scoped to the
    /// authenticated `user_id` (the `/entity` partition), each write is timed and
    /// reported live, and a per-delta failure is reported (and logged) without
    /// aborting the others.
    fn persist_planned_connections(
        &self,
        reporter: &mut TurnReporter,
        writer: &WriteDataController,
        user_id: &str,
        now_rfc3339: &str,
        audit: &crate::push_data_controller::ActionAudit,
    ) -> Vec<(WriteStatus, WriteTiming)> {
        crate::push_data_controller::planned_connection_deltas(audit)
            .into_iter()
            .map(|(change_amount, note)| {
                let start = Instant::now();
                let result =
                    writer.append_connection_delta(user_id, now_rfc3339, change_amount, &note);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let status = match result {
                    Ok(outcome) => {
                        eprintln!(
                            "recorded connection delta: {} ({:+}, balance {}, {:.1} ms)",
                            outcome.id, outcome.change_amount, outcome.new_balance, ms,
                        );
                        WriteStatus {
                            ok: true,
                            detail: format!(
                                "saved to Cosmos GaiaConnections: {} ({:+}, balance {}, {:.1} ms)",
                                outcome.id, outcome.change_amount, outcome.new_balance, ms,
                            ),
                            operations: Vec::new(),
                        }
                    }
                    Err(err) => {
                        let detail = format!("Cosmos write to GaiaConnections failed: {err}");
                        eprintln!("connection write failed for user {user_id}: {detail}");
                        WriteStatus {
                            ok: false,
                            detail,
                            operations: Vec::new(),
                        }
                    }
                };
                reporter.persist_write(&status, ms);
                let timing = WriteTiming {
                    label: "GaiaConnections delta".to_string(),
                    ms,
                    ok: status.ok,
                };
                (status, timing)
            })
            .collect()
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
    fn check_health_skips_unconfigured_dependencies_and_is_ready() {
        // A skeleton engine has no clients, so every dependency is skipped and
        // the service is ready (skipped dependencies never make it unready).
        let engine = Engine::new(None, None);
        let report = engine.check_health();
        assert!(report.ready);
        assert_eq!(report.checks.len(), 4);
        assert!(report
            .checks
            .iter()
            .all(|check| !check.configured && check.status == crate::health::CheckStatus::Skipped));
    }

    #[test]
    fn thought_ids_are_unique() {
        let a = new_thought_id();
        let b = new_thought_id();
        assert_ne!(a, b);
    }

    #[test]
    fn run_turn_reported_streams_skeleton_events_in_order() {
        // Even with no model configured, a turn must stream a live process-log
        // so the UI always knows where Gaia is. The sink receives each event as
        // it happens, and the same events are returned on the result.
        let engine = Engine::new(None, None);
        let mut streamed = Vec::new();
        let result =
            engine.run_turn_reported("alice", "hello", &mut |event| streamed.push(event.clone()));

        // The skeleton path emits exactly one event, on the `turn` phase.
        assert_eq!(streamed.len(), 1, "skeleton turn should emit one event");
        assert_eq!(streamed[0].phase, "turn");
        assert_eq!(streamed[0].level, "info");
        assert_eq!(streamed[0].seq, 0, "the first event is sequence zero");

        // The streamed events and the result's copy must match exactly.
        assert_eq!(result.events, streamed, "result must carry the same events");
    }

    #[test]
    fn persist_turn_reports_a_visible_error_without_an_online_writer() {
        // Writes are mandatory: the skeleton engine has no writer configured, so
        // persisting a turn must not silently skip. It returns a visible failure
        // status (never panics) the front end can show to the user.
        let engine = Engine::new(None, None);
        let mut events = Vec::new();
        // Bind the sink closure to a `let` so it outlives the reporter borrow.
        let mut sink = |event: &TurnEvent| events.push(event.clone());
        let mut reporter = TurnReporter::new(&mut sink);
        let status = engine.persist_turn(
            &mut reporter,
            "alice",
            "hello",
            "hi alice",
            "2026-06-27T10:00:00Z",
            &crate::push_data_controller::ActionAudit::default(),
        );
        assert!(!status.ok, "missing writer must report a failed write");
        assert!(
            status.detail.to_lowercase().contains("cosmos"),
            "detail should name Cosmos, got: {}",
            status.detail
        );
        // The failure is surfaced live as a `persist` warning, not swallowed.
        assert!(
            events
                .iter()
                .any(|e: &TurnEvent| e.phase == "persist" && e.level == "warn"),
            "a missing writer must emit a persist warning event"
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

    /// Like [`completion`], but also includes the `model` field the Azure
    /// model-router echoes to report the underlying model it selected.
    fn completion_with_model(content: &str, model: &str) -> String {
        let escaped = content.replace('\\', "\\\\").replace('"', "\\\"");
        format!(r#"{{"model":"{model}","choices":[{{"message":{{"content":"{escaped}"}}}}]}}"#)
    }

    #[test]
    fn run_turn_surfaces_the_underlying_router_model_in_debug() {
        // When the (router) backend reports which underlying model actually ran,
        // the pull/push debug panels must show that model — not the configured
        // `model-router`/`gpt-test` deployment name. `routing` keeps the
        // deployment label.
        let (endpoint, handle) = crate::test_http::spawn_mock_http_sequence(vec![
            (
                "200 OK".to_string(),
                completion_with_model("call-1 analysis", "gpt-4.1-2025-04-14"),
            ),
            (
                "200 OK".to_string(),
                completion_with_model("Final answer.", "gpt-5-mini-2025-08-07"),
            ),
        ]);
        let llm = LlmClient::for_test(endpoint);
        let engine = Engine::new(Some(llm), None);

        let result = engine.run_turn("alice", "what's up?");

        // Routing still names the configured deployment.
        assert_eq!(result.routing, "gpt-test");
        // Each debug panel reports the underlying model that ran that call.
        assert_eq!(
            result.pull_debug.expect("pull debug present").model,
            "gpt-4.1-2025-04-14"
        );
        assert_eq!(
            result.push_debug.expect("push debug present").model,
            "gpt-5-mini-2025-08-07"
        );
        handle.join().expect("mock server thread joins");
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
