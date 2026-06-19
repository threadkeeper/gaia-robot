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

use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt, Call2Prompt};
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
}

/// Runs Gaia turns. Holds the (optional) model and web-search clients plus the
/// default user id, and is cheap to clone/share across connection threads.
#[derive(Debug, Clone)]
pub struct Engine {
    /// The chat model client, or `None` to run in skeleton mode.
    llm: Option<LlmClient>,
    /// The web-search client, or `None` to skip web search.
    web_search: Option<BraveClient>,
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
        let (llm, warning) = match LlmClient::from_env() {
            Ok(client) => (client, None),
            Err(err) => (None, Some(err.to_string())),
        };
        // Web search is only meaningful alongside a model; mirror the console
        // app and only wire it when the LLM is active.
        let web_search = if llm.is_some() {
            BraveClient::from_env()
        } else {
            None
        };
        (Engine { llm, web_search }, warning)
    }

    /// Construct an engine directly from its parts (used by tests).
    #[cfg(test)]
    pub fn new(llm: Option<LlmClient>, web_search: Option<BraveClient>) -> Self {
        Engine { llm, web_search }
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
                format!(
                    "live model {} at {} ({web})",
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

        // --- Web search: assemble this turn's Response Data Context ---------
        let mut searches = Vec::new();
        let mut evidence = String::new();
        if let Some(brave) = &self.web_search {
            if !input.is_empty() {
                searches.push(input.to_string());
                match brave.search(input, 0) {
                    Ok(results) => evidence.push_str(&format_web_results(input, &results)),
                    Err(err) => {
                        evidence.push_str(&format!("Web search for \"{input}\" failed: {err}\n"))
                    }
                }
            }
        }

        // The Response Data Context handed to Call 2 = the retrieved evidence
        // plus Call 1's own reasoning about what it wanted to find.
        let response_data_context =
            format!("{evidence}\nCall 1 research plan (raw model output):\n{call1_raw}");

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
            },
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

/// Format Brave web-search results as a compact evidence block for Call 2.
fn format_web_results(query: &str, results: &[crate::search_history::SearchResult]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "Web search results for \"{query}\":");
    if results.is_empty() {
        let _ = writeln!(out, "(no results)");
    } else {
        for (i, r) in results.iter().enumerate() {
            let _ = writeln!(out, "{}. {} — {}", i + 1, r.title, r.url);
            if !r.snippet.is_empty() {
                let _ = writeln!(out, "   {}", r.snippet);
            }
        }
    }
    out
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
    fn thought_ids_are_unique() {
        let a = new_thought_id();
        let b = new_thought_id();
        assert_ne!(a, b);
    }
}
