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

use crate::actions::{ActionPlan, ActionsFile, SessionContext};
use crate::cosmos::CosmosClient;
use crate::embeddings::EmbeddingClient;
use crate::executor::Executor;
use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt, Call2Prompt};
use crate::response_context::{build_response_data_context, parse_call1_extras, RetrievalGroup};
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
            Ok(call2_raw) => TurnResult {
                reply: extract_reply_text(&call2_raw),
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
                actions_summary: summarize_call2_actions(&call2_raw),
            },
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
        // Call 1's non-action documents always parse (degrading to defaults).
        let extras = parse_call1_extras(call1_raw);

        let mut groups: Vec<RetrievalGroup> = Vec::new();
        let mut searches: Vec<String> = Vec::new();

        // Parse the action plan; without one we still emit a deterministic (but
        // result-free) context from the extras alone.
        if let Some(mut actions) = crate::actions::parse_call1_actions(call1_raw) {
            // Honour the GAIA_FORCE_SEMANTIC override exactly as the self-test does.
            if crate::executor::force_semantic() {
                crate::executor::force_semantic_on(&mut actions);
            }

            // Split the plan into Cosmos-backed queries and Web searches.
            let (web_actions, cosmos_actions): (Vec<ActionPlan>, Vec<ActionPlan>) = actions
                .actions
                .into_iter()
                .partition(|action| action.target.eq_ignore_ascii_case("Web"));

            // --- Cosmos retrieval ---------------------------------------
            if !cosmos_actions.is_empty() {
                if let Some(cosmos) = &self.cosmos {
                    // Preserve the action id + target alongside each outcome.
                    let targets: Vec<String> =
                        cosmos_actions.iter().map(|a| a.target.clone()).collect();
                    let action_ids: Vec<String> =
                        cosmos_actions.iter().map(|a| a.id.clone()).collect();
                    let plan = ActionsFile {
                        version: actions.version.clone(),
                        session: SessionContext {
                            user_id: user_id.to_string(),
                            requested_at: requested_at.to_string(),
                        },
                        actions: cosmos_actions,
                    };
                    let outcomes = match &self.embedder {
                        Some(embedder) => {
                            Executor::with_embedder(cosmos, Some(embedder.clone())).run(&plan)
                        }
                        None => Executor::new(cosmos).run(&plan),
                    };
                    for ((target, action_id), outcome) in
                        targets.iter().zip(action_ids.iter()).zip(outcomes.iter())
                    {
                        if let Ok(records) = &outcome.result {
                            let records: Vec<serde_json::Value> = records
                                .iter()
                                .filter_map(|r| serde_json::to_value(r).ok())
                                .collect();
                            groups.push(RetrievalGroup {
                                action_id: action_id.clone(),
                                container: target.clone(),
                                records,
                            });
                        }
                    }
                }
            }

            // --- Web (Brave) retrieval ----------------------------------
            if !web_actions.is_empty() {
                if let Some(brave) = &self.web_search {
                    for action in &web_actions {
                        if action.validate().is_err() {
                            continue;
                        }
                        let query = web_query_for(action, input);
                        if let Ok(results) = brave.search(&query, action.effective_top()) {
                            searches.push(query);
                            let records: Vec<serde_json::Value> = results
                                .iter()
                                .filter_map(|r| serde_json::to_value(r).ok())
                                .collect();
                            groups.push(RetrievalGroup {
                                action_id: action.id.clone(),
                                container: "Web".to_string(),
                                records,
                            });
                        }
                    }
                }
            }
        }

        let context = build_response_data_context(user_id, input, requested_at, &extras, &groups);
        (context, searches)
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

/// Choose the query string for a `Web` action.
///
/// The model's free-text `text` filter is the most precise signal, then its
/// `intent`; if neither is present we fall back to the user's input so a Web
/// action always has *something* to search for. Mirrors the self-test's
/// `web_query_for` so the cloud app and the probe behave identically.
fn web_query_for(action: &ActionPlan, input: &str) -> String {
    if let Some(text) = action
        .filters
        .text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        return text.to_string();
    }
    let intent = action.intent.trim();
    if !intent.is_empty() {
        return intent.to_string();
    }
    input.trim().to_string()
}

/// Summarize the side effects LLM Call 2 planned in its `actions.json`.
///
/// Call 2 emits `[response.json, actions.json]`; this reads the second document,
/// audits it with the exact same classifier the push-pass self-test uses
/// ([`crate::data_execution::audit_actions`]), and renders a short multi-line
/// summary ([`crate::data_execution::summarize_actions`]). Returns `None` when
/// the reply has no parseable actions document or planned nothing actionable, so
/// the [`TurnResult::actions_summary`] field is simply omitted.
fn summarize_call2_actions(raw: &str) -> Option<String> {
    let cleaned = strip_code_fences(raw.trim());
    let documents = crate::actions::extract_call1_array(cleaned)?;
    let actions = documents.get(1)?;
    let audit = crate::data_execution::audit_actions(actions);
    let summary = crate::data_execution::summarize_actions(&audit);
    if summary.trim().is_empty() {
        None
    } else {
        Some(summary)
    }
}

/// Pull Gaia's reply text out of LLM Call 2's raw output.
///
/// Call 2 is asked to emit `[response.json, actions.json]`. We try to honour
/// that structure: parse the (fence-stripped) output as JSON and read
/// `response.json.text`. If the model strayed from the format, we fall back to
/// the raw text so the user still sees *something* rather than an empty bubble.
fn extract_reply_text(raw: &str) -> String {
    let cleaned = strip_code_fences(raw.trim());

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(cleaned) {
        // Preferred shape: a JSON array whose first element is response.json.
        if let Some(text) = value
            .as_array()
            .and_then(|a| a.first())
            .and_then(|first| first.get("text"))
            .and_then(|t| t.as_str())
        {
            return text.trim().to_string();
        }
        // Tolerate a bare response.json object: { "text": ... }.
        if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
            return text.trim().to_string();
        }
    }

    // The model did not return parseable JSON; show its raw text.
    cleaned.to_string()
}

/// Strip a leading/trailing Markdown code fence (```/```json) if present.
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // Drop an optional language tag on the opening fence's line.
    let after_lang = match after_open.split_once('\n') {
        Some((_lang, rest)) => rest,
        None => after_open,
    };
    after_lang
        .trim_end()
        .strip_suffix("```")
        .unwrap_or(after_lang)
        .trim()
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
    fn extracts_text_from_response_array() {
        let raw = r#"[{"text":"Hi there","emote":"warm","medium":"console"},{"version":"1.0"}]"#;
        assert_eq!(extract_reply_text(raw), "Hi there");
    }

    #[test]
    fn extracts_text_from_bare_object() {
        assert_eq!(extract_reply_text(r#"{"text":"hello"}"#), "hello");
    }

    #[test]
    fn strips_code_fences_before_parsing() {
        let raw = "```json\n[{\"text\":\"fenced\"}]\n```";
        assert_eq!(extract_reply_text(raw), "fenced");
    }

    #[test]
    fn falls_back_to_raw_text_when_not_json() {
        assert_eq!(extract_reply_text("just words"), "just words");
    }

    #[test]
    fn summarizes_call2_actions_from_the_second_document() {
        // A well-formed [response.json, actions.json] reply yields a summary that
        // names each planned side effect.
        let raw = r#"[
          { "text": "Hello" },
          { "actions": [
            { "id": "a1", "kind": "send", "target": "WhatsApp",
              "to_name": "Jonty", "message": "Hi", "urgency": 0.7 },
            { "id": "a4", "kind": "upsert", "target": "GaiaDiary", "payload": {} }
          ] }
        ]"#;
        let summary = summarize_call2_actions(raw).expect("a summary");
        assert!(summary.contains("WhatsApp to Jonty: sent"));
        assert!(summary.contains("Saved to: GaiaDiary"));
    }

    #[test]
    fn summarizes_call2_actions_returns_none_without_actions() {
        // Only response.json present (no second element): nothing to summarize.
        assert!(summarize_call2_actions(r#"[{ "text": "Hi" }]"#).is_none());
        // Not parseable as a JSON array at all.
        assert!(summarize_call2_actions("just words").is_none());
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
