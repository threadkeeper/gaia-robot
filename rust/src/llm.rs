//! The [`LlmClient`] type: a minimal client for Gaia's two "LLM Call" blocks.
//!
//! In **dev / local mode** Gaia talks to one of two chat-completions backends,
//! chosen automatically from the environment:
//!
//! - **Azure Foundry model-router** (preferred). When `FOUNDRY_ENDPOINT` and
//!   `MODEL_ROUTER_DEPLOYMENT` are set, the client targets the same
//!   `model-router` deployment the deployed Container App calls, so dev and prod
//!   exercise the identical model. Authentication uses the Foundry **API key**
//!   from `FOUNDRY_API_KEY` by default (sent in the Azure OpenAI `api-key`
//!   header), with an Azure AD bearer token from `FOUNDRY_AAD_TOKEN` as a
//!   fallback (mint one with
//!   `az account get-access-token --resource https://cognitiveservices.azure.com`).
//! - **GitHub Models** (fallback, `gpt-4o-mini` by default). Used when Foundry
//!   is not configured, so the LLM steps still run without any Azure resources.
//!   Authenticated with a GitHub token.
//!
//! The mode is **opt-in** so the default skeleton behaviour (and the CLI tests)
//! stay unchanged: set `GAIA_MODE=dev` (or `local`) to enable live calls.
//! Without it, [`LlmClient::from_env`] returns `Ok(None)` and `main` keeps just
//! logging each block.
//!
//! Configuration is read from the environment, falling back to a local `.env`
//! file for convenience (so the values in `infra/.env` "just work" in dev):
//! - `GITHUB_TOKEN` — GitHub Models auth token (required for the fallback).
//! - `FOUNDRY_ENDPOINT` — Foundry account endpoint, e.g.
//!   `https://<account>.cognitiveservices.azure.com/`.
//! - `MODEL_ROUTER_DEPLOYMENT` — the chat deployment name (e.g. `model-router`).
//! - `FOUNDRY_API_KEY` — Foundry API key (preferred auth, `api-key` header).
//! - `FOUNDRY_AAD_TOKEN` — Azure AD bearer token (fallback auth) for the Foundry call.
//! - Managed identity (SAMI/UAMI) is auto-attempted when no key/token env var is
//!   set and Azure identity endpoint variables are present in the environment.
//! - `FOUNDRY_API_VERSION` — override the Azure OpenAI data-plane API version.
//! - `GAIA_LLM_ENDPOINT` — override the GitHub Models chat-completions URL.
//! - `GAIA_LLM_MODEL` — override the GitHub Models model name.
//! - `GAIA_ENV_FILE` — explicit path to a `.env` file to read values from.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Default GitHub Models chat-completions endpoint.
const DEFAULT_ENDPOINT: &str = "https://models.github.ai/inference/chat/completions";
/// Default model used for dev/local testing.
const DEFAULT_MODEL: &str = "gpt-4o-mini";
/// Default Azure OpenAI data-plane API version used for the Foundry
/// model-router chat call. Overridable with `FOUNDRY_API_VERSION`.
const DEFAULT_FOUNDRY_API_VERSION: &str = "2024-10-21";
/// Default GitHub user id used for user isolation in dev/local mode when
/// `GAIA_USER_ID` is not set. Matches the `threadkeeper` exports under
/// `migrations/`, so dev runs are scoped to that user's real data out of the box.
const DEFAULT_USER_ID: &str = "threadkeeper";
/// Cap on response length. Set to the model-router's maximum output token
/// limit (32768) so we don't truncate completions; the router bills only for
/// tokens actually produced, so a high ceiling costs nothing on short replies.
const DEFAULT_MAX_TOKENS: u32 = 32_768;

/// Errors that can occur while configuring or calling the LLM.
#[derive(Debug)]
pub enum LlmError {
    /// Dev mode is enabled but no GitHub token could be found.
    MissingToken,
    /// The Foundry model-router is configured (endpoint + deployment) but no
    /// credential (API key or Azure AD token) was found to authenticate the call.
    MissingFoundryToken,
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
            LlmError::MissingFoundryToken => write!(
                f,
                "no Azure Foundry credential found (set FOUNDRY_API_KEY, or an \
                 Azure AD token in FOUNDRY_AAD_TOKEN, e.g. \
                 `az account get-access-token --resource https://cognitiveservices.azure.com`)"
            ),
            LlmError::Http(msg) => write!(f, "LLM request failed: {msg}"),
            LlmError::Decode(msg) => write!(f, "could not decode LLM response: {msg}"),
            LlmError::EmptyResponse => write!(f, "LLM returned an empty response"),
        }
    }
}

impl std::error::Error for LlmError {}

/// How a credential is presented to the chat-completions endpoint.
///
/// Azure Foundry / Azure OpenAI accepts an API key in a dedicated `api-key`
/// header, while GitHub Models (and Foundry's Azure AD fallback) use the
/// standard `Authorization: Bearer` header. The client records which scheme its
/// token uses so [`LlmClient::complete`] sets the correct header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthScheme {
    /// Azure-style `api-key: <key>` header (Foundry API-key auth, the default).
    ApiKey,
    /// Standard `Authorization: Bearer <token>` header (GitHub Models, AAD).
    Bearer,
}

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
    /// Secret used to authenticate (an API key or a bearer token).
    token: String,
    /// Which HTTP header carries [`Self::token`].
    auth: AuthScheme,
}

impl LlmClient {
    /// Build a client from the environment, or return `Ok(None)` when dev/local
    /// LLM mode is not enabled.
    ///
    /// When dev mode is on, the Azure Foundry model-router is preferred (so dev
    /// and prod call the same model); it is used whenever `FOUNDRY_ENDPOINT` and
    /// `MODEL_ROUTER_DEPLOYMENT` are configured. Otherwise the client falls back
    /// to GitHub Models for a zero-Azure dev loop.
    ///
    /// Returns:
    /// - `Ok(None)` when `GAIA_MODE` is not `dev`/`local` (skeleton behaviour).
    /// - `Ok(Some(client))` when dev mode is on and a backend was resolved.
    /// - `Err(LlmError::MissingFoundryToken)` when Foundry is configured but no
    ///   Azure AD token was found.
    /// - `Err(LlmError::MissingToken)` when falling back to GitHub Models but no
    ///   GitHub token exists.
    pub fn from_env() -> Result<Option<Self>, LlmError> {
        if !dev_mode_enabled() {
            return Ok(None);
        }

        // Prefer the Azure Foundry model-router when it is configured. This is
        // the same deployment the deployed Container App calls.
        if let Some(client) = Self::foundry_from_env()? {
            return Ok(Some(client));
        }

        // Otherwise fall back to GitHub Models (token-only dev setups).
        let token = resolve_token().ok_or(LlmError::MissingToken)?;
        let endpoint =
            std::env::var("GAIA_LLM_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        let model = std::env::var("GAIA_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        Ok(Some(Self {
            endpoint,
            model,
            token,
            // GitHub Models uses the standard bearer header.
            auth: AuthScheme::Bearer,
        }))
    }

    /// Build a client targeting the Azure Foundry model-router, or `Ok(None)`
    /// when Foundry is not configured for this dev run.
    ///
    /// Foundry is selected when both `FOUNDRY_ENDPOINT` and
    /// `MODEL_ROUTER_DEPLOYMENT` are present. When they are, a credential is
    /// required: the `FOUNDRY_API_KEY` is preferred (sent in the `api-key`
    /// header), falling back to an Azure AD token in `FOUNDRY_AAD_TOKEN` (sent
    /// as a bearer token). If neither is set the call returns
    /// `Err(MissingFoundryToken)`. The chat-completions URL is built from the
    /// endpoint, the deployment name, and the (overridable) API version; the
    /// deployment name doubles as the `model` field sent in the request body.
    fn foundry_from_env() -> Result<Option<Self>, LlmError> {
        let api_version = value_from_env("FOUNDRY_API_VERSION")
            .unwrap_or_else(|| DEFAULT_FOUNDRY_API_VERSION.to_string());

        let config = resolve_foundry_config(
            value_from_env("FOUNDRY_ENDPOINT"),
            value_from_env("MODEL_ROUTER_DEPLOYMENT"),
            api_version,
            value_from_env("FOUNDRY_API_KEY"),
            value_from_env("FOUNDRY_AAD_TOKEN"),
        )?;

        Ok(config.map(|(endpoint, model, token, auth)| Self {
            endpoint,
            model,
            token,
            auth,
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

        // Choose the authentication header based on the credential scheme:
        // Foundry API keys go in the Azure `api-key` header, while bearer tokens
        // (GitHub Models, Foundry AAD) use the standard `Authorization` header.
        let request = ureq::post(&self.endpoint).set("Content-Type", "application/json");
        let request = match self.auth {
            AuthScheme::ApiKey => request.set("api-key", &self.token),
            AuthScheme::Bearer => request.set("Authorization", &format!("Bearer {}", self.token)),
        };

        let response = request.send_bytes(&payload).map_err(map_ureq_error)?;

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

/// Decide the Foundry client configuration from resolved environment values.
///
/// Returns:
/// - `Ok(None)` when Foundry is not configured (no endpoint or no deployment),
///   so the caller can fall back to GitHub Models.
/// - `Ok(Some((url, deployment, token, auth)))` when fully configured. The
///   `api_key` is preferred (presented via the `api-key` header); an Azure AD
///   token in `aad_token` is the fallback (presented as a bearer token).
/// - `Err(LlmError::MissingFoundryToken)` when the endpoint and deployment are
///   set but neither a key nor a token was found.
///
/// Splitting the pure decision out of [`LlmClient::foundry_from_env`] keeps the
/// branching logic easy to test without touching the process environment.
fn resolve_foundry_config(
    endpoint: Option<String>,
    deployment: Option<String>,
    api_version: String,
    api_key: Option<String>,
    aad_token: Option<String>,
) -> Result<Option<(String, String, String, AuthScheme)>, LlmError> {
    // Foundry only kicks in when we know *where* to call and *which* deployment.
    let (Some(endpoint), Some(deployment)) = (endpoint, deployment) else {
        return Ok(None);
    };

    // Once Foundry is selected, a credential is mandatory: failing loudly is
    // better than silently falling back to a different model. Prefer the API
    // key (the default for local dev), then the Azure AD token.
    let (token, auth) = if let Some(key) = api_key {
        (key, AuthScheme::ApiKey)
    } else if let Some(aad) = aad_token {
        (aad, AuthScheme::Bearer)
    } else if let Some(mi) = managed_identity_token("https://cognitiveservices.azure.com") {
        (mi, AuthScheme::Bearer)
    } else {
        return Err(LlmError::MissingFoundryToken);
    };

    let url = foundry_chat_url(&endpoint, &deployment, &api_version);
    Ok(Some((url, deployment, token, auth)))
}

/// Build the Azure OpenAI chat-completions URL for a Foundry deployment.
///
/// The shape is
/// `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={ver}`.
/// Any trailing slash on `endpoint` is trimmed so we never produce a double `//`.
fn foundry_chat_url(endpoint: &str, deployment: &str, api_version: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    format!("{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}")
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

/// Fetch a bearer token for `resource` from the local Azure managed identity
/// endpoint, if one is available in this environment.
///
/// This is used to support cloud runtimes (for example Container Apps with
/// system-assigned managed identity) without requiring `FOUNDRY_API_KEY` or
/// a pre-minted `FOUNDRY_AAD_TOKEN` in environment variables.
pub fn managed_identity_token(resource: &str) -> Option<String> {
    let mut resources = vec![resource.to_string()];
    if !resource.ends_with('/') {
        resources.push(format!("{resource}/"));
    }
    resources.push("https://cognitiveservices.azure.com/.default".to_string());

    // Newer endpoint contract used by Container Apps/App Service.
    if let (Ok(endpoint), Ok(header)) = (
        std::env::var("IDENTITY_ENDPOINT"),
        std::env::var("IDENTITY_HEADER"),
    ) {
        for api_version in ["2019-08-01", "2017-09-01"] {
            for resource_value in &resources {
                let url = format!(
                    "{}?api-version={api_version}&resource={}",
                    endpoint,
                    percent_encode_query_value(resource_value)
                );
                if let Ok(response) = ureq::get(&url).set("X-IDENTITY-HEADER", &header).call() {
                    if let Ok(body) = response.into_string() {
                        if let Some(token) = parse_managed_identity_access_token(&body) {
                            return Some(token);
                        }
                    }
                }
            }
        }
    }

    // Legacy MSI endpoint contract.
    if let (Ok(endpoint), Ok(secret)) = (std::env::var("MSI_ENDPOINT"), std::env::var("MSI_SECRET"))
    {
        for api_version in ["2019-08-01", "2017-09-01"] {
            for resource_value in &resources {
                let url = format!(
                    "{}?api-version={api_version}&resource={}",
                    endpoint,
                    percent_encode_query_value(resource_value)
                );
                if let Ok(response) = ureq::get(&url).set("secret", &secret).call() {
                    if let Ok(body) = response.into_string() {
                        if let Some(token) = parse_managed_identity_access_token(&body) {
                            return Some(token);
                        }
                    }
                }
            }
        }
    }

    // IMDS fallback used by many Azure hosts when custom endpoint vars are not
    // present in the process environment.
    for resource_value in &resources {
        let imds_url = format!(
            "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource={}",
            percent_encode_query_value(resource_value)
        );
        if let Ok(response) = ureq::get(&imds_url).set("Metadata", "true").call() {
            if let Ok(body) = response.into_string() {
                if let Some(token) = parse_managed_identity_access_token(&body) {
                    return Some(token);
                }
            }
        }
    }

    None
}

/// Parse an access token out of a managed-identity token response body.
fn parse_managed_identity_access_token(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .get("access_token")
        .or_else(|| value.get("accessToken"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
}

/// Percent-encode one query-string value per RFC 3986.
fn percent_encode_query_value(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            _ => {
                encoded.push('%');
                encoded.push(hex_digit(byte >> 4));
                encoded.push(hex_digit(byte & 0x0f));
            }
        }
    }
    encoded
}

/// Convert a nibble to uppercase hex.
fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + (nibble - 10)) as char,
    }
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
    fn foundry_chat_url_builds_the_azure_openai_path() {
        let expected = "https://acct.cognitiveservices.azure.com/openai/deployments/\
                        model-router/chat/completions?api-version=2024-10-21";
        // A trailing slash on the endpoint must not produce a double `//`.
        assert_eq!(
            foundry_chat_url(
                "https://acct.cognitiveservices.azure.com/",
                "model-router",
                "2024-10-21"
            ),
            expected
        );
        // And it works identically without the trailing slash.
        assert_eq!(
            foundry_chat_url(
                "https://acct.cognitiveservices.azure.com",
                "model-router",
                "2024-10-21"
            ),
            expected
        );
    }

    #[test]
    fn resolve_foundry_config_returns_none_when_not_configured() {
        // Missing deployment -> not configured.
        let none_deployment = resolve_foundry_config(
            Some("https://acct.cognitiveservices.azure.com/".to_string()),
            None,
            "2024-10-21".to_string(),
            Some("key".to_string()),
            Some("token".to_string()),
        )
        .unwrap();
        assert!(none_deployment.is_none());

        // Missing endpoint -> not configured.
        let none_endpoint = resolve_foundry_config(
            None,
            Some("model-router".to_string()),
            "2024-10-21".to_string(),
            Some("key".to_string()),
            Some("token".to_string()),
        )
        .unwrap();
        assert!(none_endpoint.is_none());
    }

    #[test]
    fn resolve_foundry_config_requires_a_credential_when_configured() {
        let err = resolve_foundry_config(
            Some("https://acct.cognitiveservices.azure.com/".to_string()),
            Some("model-router".to_string()),
            "2024-10-21".to_string(),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, LlmError::MissingFoundryToken));
    }

    #[test]
    fn resolve_foundry_config_prefers_the_api_key() {
        // When both are present, the API key wins and uses the `api-key` header.
        let (url, deployment, token, auth) = resolve_foundry_config(
            Some("https://acct.cognitiveservices.azure.com/".to_string()),
            Some("model-router".to_string()),
            "2024-10-21".to_string(),
            Some("the-api-key".to_string()),
            Some("aad-token".to_string()),
        )
        .unwrap()
        .expect("fully configured Foundry should yield a config");

        assert_eq!(deployment, "model-router");
        assert_eq!(token, "the-api-key");
        assert_eq!(auth, AuthScheme::ApiKey);
        assert!(url.contains("/openai/deployments/model-router/chat/completions"));
        assert!(url.contains("api-version=2024-10-21"));
    }

    #[test]
    fn resolve_foundry_config_falls_back_to_the_aad_token() {
        // With no API key, the Azure AD token is used as a bearer credential.
        let (_url, _deployment, token, auth) = resolve_foundry_config(
            Some("https://acct.cognitiveservices.azure.com/".to_string()),
            Some("model-router".to_string()),
            "2024-10-21".to_string(),
            None,
            Some("aad-token".to_string()),
        )
        .unwrap()
        .expect("a token-only Foundry config should still resolve");

        assert_eq!(token, "aad-token");
        assert_eq!(auth, AuthScheme::Bearer);
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
            auth: AuthScheme::Bearer,
        };

        let preview = client.request_preview("SYSTEM-CONTEXT", "USER-INPUT");

        // Wire parameters are shown.
        assert!(preview.contains("model        : gpt-4o-mini"));
        assert!(preview.contains("max_tokens   : 32768"));
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

    #[test]
    fn parse_managed_identity_access_token_reads_access_token_field() {
        let body = r#"{"access_token":"abc123","expires_in":"3599"}"#;
        assert_eq!(
            parse_managed_identity_access_token(body),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn parse_managed_identity_access_token_reads_access_token_camel_case_field() {
        let body = r#"{"accessToken":"abc123"}"#;
        assert_eq!(
            parse_managed_identity_access_token(body),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn percent_encode_query_value_encodes_reserved_characters() {
        assert_eq!(
            percent_encode_query_value("https://cognitiveservices.azure.com"),
            "https%3A%2F%2Fcognitiveservices.azure.com"
        );
    }
}
