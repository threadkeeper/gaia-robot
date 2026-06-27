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
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::cosmos::CosmosClient;
use crate::embeddings::EmbeddingClient;
use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt, Call2Prompt};
use crate::pull_data_controller::PullDataController;
use crate::push_data_controller::PushDataController;
use crate::web_search::BraveClient;

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
        (
            Engine {
                llm,
                web_search,
                cosmos,
                embedder,
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
                format!(
                    "live model {} at {} ({web}, {cosmos}, {embed})",
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
            };
        };

        let requested_at = now_rfc3339();

        // --- LLM Call 1: the pull / research pass --------------------------
        let call1 = Call1Prompt::build(user_id, input, "", &requested_at);
        let call1_raw = match client.complete(&call1.system, &call1.user) {
            Ok(text) => text,
            Err(err) => {
                // Call 1 failed: we can still answer, just without a research plan.
                format!("(LLM Call 1 failed: {err})")
            }
        };

        // --- Retrieval + deterministic Response Data Context ---------------
        // Execute every retrieval action Call 1 planned and fold the results,
        // together with Call 1's analysis/facts/newContext, into the eight-
        // section markdown that grounds Call 2. This is the exact same builder
        // the data-retrieval self-test uses, so the cloud app and the test stay
        // in lock-step. It never makes an extra LLM call.
        let (response_data_context, searches) =
            self.assemble_context(user_id, input, &requested_at, &call1_raw);

        // --- LLM Call 2: the push / answer pass ----------------------------
        let call2 = Call2Prompt::build(user_id, input, &response_data_context, &requested_at);
        match client.complete(&call2.system, &call2.user) {
            Ok(call2_raw) => {
                // Drive the shared push controller — the exact same parsing and
                // audit the data-execution self-test runs. It yields both the
                // reply text to show and the planned-side-effects summary bubble.
                let push = PushDataController::process(&call2_raw);
                // Capture the actions summary before moving the reply text into
                // the result struct (both borrow `push`).
                let actions_summary = push.actions_summary();
                TurnResult {
                    reply: push.reply_text,
                    verdict: "allow".to_string(),
                    routing: client.model().to_string(),
                    attention: 0.0,
                    thought_id,
                    searches: if searches.is_empty() {
                        None
                    } else {
                        Some(searches)
                    },
                    // Audit Call 2's actions.json and surface a short summary bubble.
                    actions_summary,
                }
            }
            Err(err) => TurnResult {
                reply: format!("Sorry, I could not complete my reply: {err}"),
                verdict: "error".to_string(),
                routing: client.model().to_string(),
                attention: 0.0,
                thought_id,
                searches: if searches.is_empty() {
                    None
                } else {
                    Some(searches)
                },
                actions_summary: None,
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
    /// Returns the context markdown and the web queries that actually ran (for
    /// the [`TurnResult::searches`] field).
    fn assemble_context(
        &self,
        user_id: &str,
        input: &str,
        requested_at: &str,
        call1_raw: &str,
    ) -> (String, Vec<String>) {
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
        (result.context, result.searches)
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

        let (context, searches) =
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
        let (again, _) = engine.assemble_context("alice", "hi", "2026-06-21T00:00:00Z", reply);
        assert_eq!(context, again);
    }
}
