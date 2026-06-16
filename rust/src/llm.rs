//! The [`LlmClient`] type: a minimal client for Gaia's two "LLM Call" blocks.
//!
//! In **dev / local mode** Gaia talks to the GitHub Models inference endpoint
//! (`gpt-4o-mini` by default), authenticated with a GitHub token. This lets us
//! exercise the LLM steps of the flow without provisioning any Azure resources.
//!
//! The mode is **opt-in** so the default skeleton behaviour (and the CLI tests)
//! stay unchanged: set `GAIA_MODE=dev` (or `local`) to enable live calls.
//! Without it, [`LlmClient::from_env`] returns `Ok(None)` and `main` keeps just
//! logging each block.
//!
//! Configuration is read from the environment, falling back to a local `.env`
//! file for convenience (so the token in `infra/.env` "just works" in dev):
//! - `GITHUB_TOKEN` — auth token (required when dev mode is enabled).
//! - `GAIA_LLM_ENDPOINT` — override the chat-completions URL.
//! - `GAIA_LLM_MODEL` — override the model name.
//! - `GAIA_ENV_FILE` — explicit path to a `.env` file to read the token from.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Default GitHub Models chat-completions endpoint.
const DEFAULT_ENDPOINT: &str = "https://models.github.ai/inference/chat/completions";
/// Default model used for dev/local testing.
const DEFAULT_MODEL: &str = "gpt-4o-mini";
/// Default GitHub user id used for user isolation in dev/local mode when
/// `GAIA_USER_ID` is not set. Matches the `threadkeeper` exports under
/// `migrations/`, so dev runs are scoped to that user's real data out of the box.
const DEFAULT_USER_ID: &str = "threadkeeper";
/// Conservative cap on response length so dev testing stays cheap and fast.
const DEFAULT_MAX_TOKENS: u32 = 512;

/// Errors that can occur while configuring or calling the LLM.
#[derive(Debug)]
pub enum LlmError {
    /// Dev mode is enabled but no GitHub token could be found.
    MissingToken,
    /// The HTTP request failed or returned a non-success status.
    Http(String),
    /// The response body could not be decoded into the expected shape.
    Decode(String),
    /// The response contained no usable message content.
    EmptyResponse,
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::MissingToken => write!(
                f,
                "no GitHub token found (set GITHUB_TOKEN or put it in infra/.env)"
            ),
            LlmError::Http(msg) => write!(f, "LLM request failed: {msg}"),
            LlmError::Decode(msg) => write!(f, "could not decode LLM response: {msg}"),
            LlmError::EmptyResponse => write!(f, "LLM returned an empty response"),
        }
    }
}

impl std::error::Error for LlmError {}

/// A minimal, immutable client for a single chat-completions endpoint.
///
/// Construct one with [`LlmClient::from_env`] and make calls with
/// [`LlmClient::complete`]. The client is cheap to clone and holds no network
/// state of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmClient {
    /// Full chat-completions URL to POST to.
    endpoint: String,
    /// Model name sent in the request body, e.g. `gpt-4o-mini`.
    model: String,
    /// Bearer token used for the `Authorization` header.
    token: String,
}

impl LlmClient {
    /// Build a client from the environment, or return `Ok(None)` when dev/local
    /// LLM mode is not enabled.
    ///
    /// Returns:
    /// - `Ok(None)` when `GAIA_MODE` is not `dev`/`local` (skeleton behaviour).
    /// - `Ok(Some(client))` when dev mode is on and a token was resolved.
    /// - `Err(LlmError::MissingToken)` when dev mode is on but no token exists.
    pub fn from_env() -> Result<Option<Self>, LlmError> {
        if !dev_mode_enabled() {
            return Ok(None);
        }

        let token = resolve_token().ok_or(LlmError::MissingToken)?;
        let endpoint =
            std::env::var("GAIA_LLM_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        let model = std::env::var("GAIA_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        Ok(Some(Self {
            endpoint,
            model,
            token,
        }))
    }

    /// The model name this client sends, e.g. `gpt-4o-mini`.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The chat-completions endpoint this client posts to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Send a single-turn chat completion and return the assistant's text.
    ///
    /// `system` carries Gaia's context/instructions; `user` is the end user's
    /// input. The call is blocking and returns the trimmed message content.
    pub fn complete(&self, system: &str, user: &str) -> Result<String, LlmError> {
        let body = build_request_body(&self.model, system, user, DEFAULT_MAX_TOKENS);

        // Serialize with serde_json directly (rather than ureq's optional json
        // feature) so we keep ureq's feature set minimal and reuse the crate we
        // already depend on.
        let payload = serde_json::to_vec(&body).map_err(|e| LlmError::Decode(e.to_string()))?;

        let response = ureq::post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/json")
            .send_bytes(&payload)
            .map_err(map_ureq_error)?;

        let text = response
            .into_string()
            .map_err(|e| LlmError::Http(e.to_string()))?;

        parse_completion(&text)
    }

    /// Render a human-readable preview of *exactly* what this client will send
    /// for one chat completion, so a walk-through can see the full context
    /// window with nothing hidden.
    ///
    /// The preview lists the wire parameters (endpoint, model, `max_tokens`),
    /// the set of attached tools / functions / MCP servers / skills (currently
    /// **none** — the dev client is a plain chat-completions call), and then the
    /// complete, untruncated `system` and `user` message contents with rough
    /// size estimates. The bearer token is intentionally **never** included.
    pub fn request_preview(&self, system: &str, user: &str) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();

        // Wire parameters that actually go in the request.
        // (`let _ =` because writing to a String is infallible.)
        let _ = writeln!(out, "    endpoint     : {}", self.endpoint);
        let _ = writeln!(out, "    model        : {}", self.model);
        let _ = writeln!(out, "    max_tokens   : {DEFAULT_MAX_TOKENS}");

        // Attachments. The dev client is deliberately minimal: no tool/function
        // calling, no MCP servers, no skills. We print these explicitly so the
        // walk-through shows that nothing beyond the two messages is attached.
        let _ = writeln!(out, "    tools        : none attached");
        let _ = writeln!(out, "    functions    : none attached");
        let _ = writeln!(out, "    MCP servers  : none attached");
        let _ = writeln!(out, "    skills       : none attached");
        let _ = writeln!(out, "    response_fmt : text (model default)");

        // The full context window: both messages exactly as they will be sent.
        let _ = writeln!(
            out,
            "    messages     : 2 (system + user), {} chars total / ~{} tokens",
            system.chars().count() + user.chars().count(),
            approx_tokens(system) + approx_tokens(user),
        );
        let _ = writeln!(
            out,
            "    --- [system] {} chars / ~{} tokens ---",
            system.chars().count(),
            approx_tokens(system),
        );
        let _ = writeln!(out, "{system}");
        let _ = writeln!(
            out,
            "    --- [user] {} chars / ~{} tokens ---",
            user.chars().count(),
            approx_tokens(user),
        );
        let _ = writeln!(out, "{user}");

        out
    }
}

/// True when the given flow-block title is one of the two LLM call blocks.
///
/// `main` uses this to decide which blocks should trigger a live model request
/// when dev/local mode is enabled.
pub fn is_llm_call(title: &str) -> bool {
    title == "LLM Call 1" || title == "LLM Call 2"
}

/// Resolve the GitHub user id used for user isolation in dev/local mode.
///
/// Reads `GAIA_USER_ID` from the environment or a `.env` file (same precedence
/// as the dev token), falling back to [`DEFAULT_USER_ID`]. Every read and write
/// in a dev turn should be scoped to this id so one user never sees another's
/// data.
pub fn dev_user_id() -> String {
    value_from_env("GAIA_USER_ID").unwrap_or_else(|| DEFAULT_USER_ID.to_string())
}

/// Whether dev/local LLM mode is enabled via the `GAIA_MODE` environment var.
fn dev_mode_enabled() -> bool {
    match std::env::var("GAIA_MODE") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            value == "dev" || value == "local"
        }
        Err(_) => false,
    }
}

/// Resolve the GitHub token from the environment, falling back to a `.env` file.
fn resolve_token() -> Option<String> {
    value_from_env("GITHUB_TOKEN")
}

/// Resolve a configuration value from the process environment, falling back to
/// the same `.env` files used for the dev token.
///
/// Lookup order: the real process environment variable `key` first, then the
/// candidate `.env` files (the `GAIA_ENV_FILE` override, then `infra/.env`, then
/// `../infra/.env`). Returns `None` when `key` is absent or empty everywhere.
///
/// This is the shared entry point for reading dev/local configuration (the
/// GitHub token, the dev user id, and so on) so every value honours the same
/// precedence rules.
pub fn value_from_env(key: &str) -> Option<String> {
    // Prefer a real process environment variable.
    if let Ok(value) = std::env::var(key) {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Some(value);
        }
    }

    // Fall back to a local .env file for developer convenience.
    value_from_env_files(key)
}

/// Look for `key` in the candidate `.env` file locations.
fn value_from_env_files(key: &str) -> Option<String> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    // Explicit override always wins.
    if let Ok(path) = std::env::var("GAIA_ENV_FILE") {
        candidates.push(PathBuf::from(path));
    }
    // Common locations relative to where the binary is typically launched
    // (repo root, or the `rust/` crate directory during `cargo run`).
    candidates.push(PathBuf::from("infra/.env"));
    candidates.push(PathBuf::from("../infra/.env"));

    for path in candidates {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Some(value) = parse_dotenv(&contents).get(key) {
                let value = value.trim().to_string();
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
    }

    None
}

/// Parse a minimal `.env` file into key/value pairs.
///
/// Supports `KEY=VALUE` lines, `#` comments, blank lines, and optional
/// surrounding single or double quotes around the value. This is intentionally
/// tiny — just enough to read the dev token without a `dotenv` dependency.
fn parse_dotenv(contents: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();

    for raw_line in contents.lines() {
        let line = raw_line.trim();

        // Skip blanks and comments.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split on the first '=' only; everything after it is the value.
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let key = key.trim().to_string();
        let value = strip_quotes(value.trim()).to_string();

        if !key.is_empty() {
            map.insert(key, value);
        }
    }

    map
}

/// Remove a single matching pair of surrounding single or double quotes.
fn strip_quotes(value: &str) -> &str {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}

/// Rough token estimate for a string (~4 characters per token).
///
/// This is a heuristic, not a real tokenizer: it only feeds the size figures in
/// [`LlmClient::request_preview`], mirroring the program's "approximate tokens"
/// budgeting elsewhere in the flow.
fn approx_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// Build the chat-completions request body for one user turn.
fn build_request_body<'a>(
    model: &'a str,
    system: &'a str,
    user: &'a str,
    max_tokens: u32,
) -> ChatRequest<'a> {
    ChatRequest {
        model,
        max_tokens,
        messages: vec![
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: user,
            },
        ],
    }
}

/// Extract the assistant's message text from a chat-completions JSON body.
fn parse_completion(body: &str) -> Result<String, LlmError> {
    let parsed: ChatResponse =
        serde_json::from_str(body).map_err(|e| LlmError::Decode(e.to_string()))?;

    let content = parsed
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .unwrap_or_default();

    let content = content.trim().to_string();
    if content.is_empty() {
        Err(LlmError::EmptyResponse)
    } else {
        Ok(content)
    }
}

/// Convert a `ureq` error into our own [`LlmError`], including the HTTP body for
/// non-success statuses so failures are easy to diagnose.
fn map_ureq_error(err: ureq::Error) -> LlmError {
    match err {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            LlmError::Http(format!("HTTP {code}: {body}"))
        }
        ureq::Error::Transport(transport) => LlmError::Http(transport.to_string()),
    }
}

/// The chat-completions request body we send to the endpoint.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
}

/// One message in the chat request.
#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// The subset of the chat-completions response we care about.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

/// One choice in the response.
#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

/// The message payload inside a choice.
#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_llm_call_matches_only_the_two_llm_blocks() {
        assert!(is_llm_call("LLM Call 1"));
        assert!(is_llm_call("LLM Call 2"));
        assert!(!is_llm_call("User"));
        assert!(!is_llm_call("response.json"));
    }

    #[test]
    fn parse_dotenv_reads_keys_skips_comments_and_strips_quotes() {
        let contents = "\
# a comment\n\
\n\
GITHUB_TOKEN=ghp_example123\n\
QUOTED=\"with spaces\"\n\
SINGLE='single quoted'\n\
COSMOS_ENDPOINT=https://example.documents.azure.com:443/\n\
# trailing comment\n";

        let map = parse_dotenv(contents);

        assert_eq!(map.get("GITHUB_TOKEN").unwrap(), "ghp_example123");
        assert_eq!(map.get("QUOTED").unwrap(), "with spaces");
        assert_eq!(map.get("SINGLE").unwrap(), "single quoted");
        assert_eq!(
            map.get("COSMOS_ENDPOINT").unwrap(),
            "https://example.documents.azure.com:443/"
        );
        // Comment lines must not become keys.
        assert!(!map.contains_key("# a comment"));
    }

    #[test]
    fn strip_quotes_only_removes_matching_pairs() {
        assert_eq!(strip_quotes("\"hello\""), "hello");
        assert_eq!(strip_quotes("'hello'"), "hello");
        assert_eq!(strip_quotes("plain"), "plain");
        // Mismatched quotes are left untouched.
        assert_eq!(strip_quotes("\"hello'"), "\"hello'");
    }

    #[test]
    fn value_from_env_reads_a_process_variable() {
        // Use a unique key so this test never collides with another's env var.
        let key = "GAIA_TEST_VALUE_FROM_ENV";
        std::env::set_var(key, "  hello  ");
        assert_eq!(value_from_env(key).as_deref(), Some("hello"));
        std::env::remove_var(key);
        assert_eq!(value_from_env(key), None);
    }

    #[test]
    fn dev_user_id_honours_the_env_override() {
        // An explicit GAIA_USER_ID takes precedence over the default.
        std::env::set_var("GAIA_USER_ID", "someone-else");
        assert_eq!(dev_user_id(), "someone-else");
        std::env::remove_var("GAIA_USER_ID");
    }

    #[test]
    fn approx_tokens_rounds_up_quarter_of_chars() {
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("a"), 1); // 1 char -> ceil(1/4) = 1
        assert_eq!(approx_tokens("abcd"), 1); // 4 chars -> 1
        assert_eq!(approx_tokens("abcde"), 2); // 5 chars -> ceil(5/4) = 2
    }

    #[test]
    fn request_preview_shows_full_window_and_no_attachments() {
        let client = LlmClient {
            endpoint: "https://example/inference".to_string(),
            model: "gpt-4o-mini".to_string(),
            token: "ghp_super_secret".to_string(),
        };

        let preview = client.request_preview("SYSTEM-CONTEXT", "USER-INPUT");

        // Wire parameters are shown.
        assert!(preview.contains("model        : gpt-4o-mini"));
        assert!(preview.contains("max_tokens   : 512"));
        // Attachments are explicitly none.
        assert!(preview.contains("tools        : none attached"));
        assert!(preview.contains("functions    : none attached"));
        assert!(preview.contains("MCP servers  : none attached"));
        assert!(preview.contains("skills       : none attached"));
        // The full, untruncated message contents are present.
        assert!(preview.contains("SYSTEM-CONTEXT"));
        assert!(preview.contains("USER-INPUT"));
        // The bearer token must NEVER appear in the preview.
        assert!(!preview.contains("ghp_super_secret"));
    }

    #[test]
    fn build_request_body_uses_system_and_user_roles() {
        let body = build_request_body("gpt-4o-mini", "be gaia", "hi", 42);
        let json = serde_json::to_string(&body).unwrap();

        assert!(json.contains("\"model\":\"gpt-4o-mini\""));
        assert!(json.contains("\"max_tokens\":42"));
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"content\":\"be gaia\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hi\""));
    }

    #[test]
    fn parse_completion_extracts_first_choice_content() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"  pong  "}}]}"#;
        assert_eq!(parse_completion(body).unwrap(), "pong");
    }

    #[test]
    fn parse_completion_errors_on_empty_content() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"   "}}]}"#;
        assert!(matches!(
            parse_completion(body),
            Err(LlmError::EmptyResponse)
        ));
    }

    #[test]
    fn parse_completion_errors_on_no_choices() {
        let body = r#"{"choices":[]}"#;
        assert!(matches!(
            parse_completion(body),
            Err(LlmError::EmptyResponse)
        ));
    }

    #[test]
    fn parse_completion_errors_on_malformed_json() {
        assert!(matches!(
            parse_completion("not json"),
            Err(LlmError::Decode(_))
        ));
    }

    #[test]
    fn llm_error_messages_are_descriptive() {
        assert!(LlmError::MissingToken.to_string().contains("token"));
        assert!(LlmError::Http("boom".into()).to_string().contains("boom"));
        assert!(LlmError::Decode("bad".into()).to_string().contains("bad"));
        assert!(LlmError::EmptyResponse.to_string().contains("empty"));
    }
}
