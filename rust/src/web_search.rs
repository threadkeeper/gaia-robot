//! The [`BraveClient`] type: Gaia's web-search applet, backed by the
//! **Brave Search API**.
//!
//! During LLM Call 1 (the *pull* pass) Gaia may decide a question needs fresh
//! public facts. The `Web` action in `actions.json` is dispatched here: we issue
//! a single Brave Web Search request and turn the JSON response into the small
//! [`search_history::SearchResult`] records the rest of the program already
//! understands. Those results are both folded into the Response Data Context
//! handed to Call 2 and appended to the [`search_history::SearchHistory`] audit
//! log.
//!
//! Configuration mirrors the other clients (`llm.rs`, `cosmos.rs`) and is read
//! from the environment, falling back to a local `.env` file via
//! [`crate::llm::value_from_env`]:
//! - `BRAVE_SEARCH_API_KEY` — the Brave subscription token (required). Without it
//!   [`BraveClient::from_env`] returns `None` and Gaia simply runs no web search.
//! - `BRAVE_SEARCH_ENDPOINT` — override the API URL. Defaults to
//!   [`DEFAULT_BRAVE_ENDPOINT`] (`https://api.search.brave.com/res/v1/web/search`),
//!   the same endpoint the deployed Container App is configured with.
//!
//! Authentication uses Brave's `X-Subscription-Token` header. The URL-building
//! and response-parsing logic lives in small pure functions so it can be unit
//! tested without a network or a live API key.

use serde::Deserialize;

use crate::search_history::SearchResult;

/// Default Brave Web Search API endpoint. Overridable with `BRAVE_SEARCH_ENDPOINT`.
const DEFAULT_BRAVE_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";

/// Default number of results to request when a caller does not specify one.
const DEFAULT_RESULT_COUNT: usize = 5;

/// Brave caps the `count` parameter for the web-search endpoint at 20.
const MAX_RESULT_COUNT: usize = 20;

/// Errors that can occur while calling the Brave Search API.
#[derive(Debug)]
pub enum WebSearchError {
    /// The HTTP request failed or returned a non-success status. The string
    /// includes the status code and body for easy diagnosis.
    Http(String),
    /// The response body could not be decoded into the expected shape.
    Decode(String),
}

impl std::fmt::Display for WebSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebSearchError::Http(msg) => write!(f, "web search request failed: {msg}"),
            WebSearchError::Decode(msg) => write!(f, "could not decode web search response: {msg}"),
        }
    }
}

impl std::error::Error for WebSearchError {}

/// A minimal, immutable client for the Brave Web Search API.
///
/// Build one with [`BraveClient::from_env`] and run queries with
/// [`BraveClient::search`]. The client is cheap to clone and holds no network
/// state of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BraveClient {
    /// The Brave Web Search endpoint to GET.
    endpoint: String,
    /// The Brave subscription token sent in the `X-Subscription-Token` header.
    api_key: String,
}

impl BraveClient {
    /// Build a client from the environment, or return `None` when no Brave API
    /// key is configured.
    ///
    /// Web search is **optional**: if `BRAVE_SEARCH_API_KEY` is absent (in the
    /// process environment or a `.env` file) this returns `None` and the caller
    /// simply skips web search rather than failing the turn. The endpoint
    /// defaults to [`DEFAULT_BRAVE_ENDPOINT`] but can be overridden with
    /// `BRAVE_SEARCH_ENDPOINT`.
    pub fn from_env() -> Option<Self> {
        // Resolve the raw values from the environment / .env, then let the pure
        // `from_parts` helper make the decision so the branching stays testable
        // without touching the process environment or the filesystem.
        from_parts(
            crate::llm::value_from_env("BRAVE_SEARCH_API_KEY"),
            crate::llm::value_from_env("BRAVE_SEARCH_ENDPOINT"),
        )
    }

    /// The Brave endpoint this client queries.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Test-only constructor: build a client pointing at an arbitrary endpoint
    /// (e.g. a localhost mock) with a throwaway key, so the real `search` path
    /// can be exercised without the process environment.
    #[cfg(test)]
    pub(crate) fn for_test(endpoint: impl Into<String>) -> Self {
        BraveClient {
            endpoint: endpoint.into(),
            api_key: "test-token".to_string(),
        }
    }

    /// Run a single web search and return up to `count` results.
    ///
    /// `count` is clamped to Brave's supported range (1..=20). The call is
    /// blocking. On success it returns the results in Brave's ranked order,
    /// mapped to [`SearchResult`]; an empty `Vec` means the search ran but
    /// matched nothing.
    pub fn search(&self, query: &str, count: usize) -> Result<Vec<SearchResult>, WebSearchError> {
        let url = build_search_url(&self.endpoint, query, count);

        let response = ureq::get(&url)
            // Brave authenticates with a subscription token, not a bearer token.
            .set("X-Subscription-Token", &self.api_key)
            .set("Accept", "application/json")
            .call()
            .map_err(map_ureq_error)?;

        let body = response
            .into_string()
            .map_err(|e| WebSearchError::Http(e.to_string()))?;

        parse_results(&body)
    }
}

/// Decide a [`BraveClient`] from already-resolved configuration values.
///
/// A key is mandatory: `api_key = None` yields `None` so the caller skips web
/// search. When a key is present, a missing endpoint falls back to
/// [`DEFAULT_BRAVE_ENDPOINT`]. Splitting this pure decision out of
/// [`BraveClient::from_env`] keeps the branching testable without reading the
/// process environment or a `.env` file (mirrors `llm::resolve_foundry_config`).
fn from_parts(api_key: Option<String>, endpoint: Option<String>) -> Option<BraveClient> {
    // Without a key there is nothing to authenticate with.
    let api_key = api_key?;
    let endpoint = endpoint.unwrap_or_else(|| DEFAULT_BRAVE_ENDPOINT.to_string());
    Some(BraveClient { endpoint, api_key })
}

/// Build the Brave Web Search request URL for one query.
///
/// The shape is `{endpoint}?q={query}&count={count}`, with `query` percent-encoded
/// for safe inclusion in the query string and `count` clamped to Brave's
/// supported range (1..=[`MAX_RESULT_COUNT`]). A `count` of 0 falls back to
/// [`DEFAULT_RESULT_COUNT`].
fn build_search_url(endpoint: &str, query: &str, count: usize) -> String {
    let count = clamp_count(count);
    let encoded_query = percent_encode(query);
    format!("{endpoint}?q={encoded_query}&count={count}")
}

/// Clamp a requested result count into Brave's supported 1..=20 range.
///
/// A `0` (the common "unset" value) becomes [`DEFAULT_RESULT_COUNT`]; anything
/// above [`MAX_RESULT_COUNT`] is capped so Brave never rejects the request.
fn clamp_count(count: usize) -> usize {
    if count == 0 {
        DEFAULT_RESULT_COUNT
    } else {
        count.min(MAX_RESULT_COUNT)
    }
}

/// Percent-encode a string for use as a URL query-string value (RFC 3986).
///
/// Unreserved characters (`A-Z a-z 0-9 - _ . ~`) pass through unchanged; every
/// other byte is escaped as `%XX`. This keeps the module std-only (no `url` or
/// `urlencoding` dependency) while still encoding spaces, `&`, `=`, and any
/// non-ASCII UTF-8 bytes safely.
fn percent_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        match byte {
            // Unreserved characters per RFC 3986 — safe to leave as-is.
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            // Everything else is escaped to its two-digit uppercase hex form.
            _ => {
                encoded.push('%');
                encoded.push(hex_digit(byte >> 4));
                encoded.push(hex_digit(byte & 0x0f));
            }
        }
    }
    encoded
}

/// Map a 0..=15 nibble to its uppercase hexadecimal ASCII digit.
fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        // 10..=15 -> 'A'..='F'.
        _ => (b'A' + (nibble - 10)) as char,
    }
}

/// Parse a Brave Web Search JSON body into our [`SearchResult`] records.
///
/// Only the `web.results[]` array is consumed; each entry's `title`, `url`, and
/// `description` become a [`SearchResult`]. Missing string fields default to
/// empty, and a response with no `web` block yields an empty `Vec` (a valid
/// "no results" outcome).
fn parse_results(body: &str) -> Result<Vec<SearchResult>, WebSearchError> {
    let parsed: BraveResponse =
        serde_json::from_str(body).map_err(|e| WebSearchError::Decode(e.to_string()))?;

    let results = parsed
        .web
        .map(|web| web.results)
        .unwrap_or_default()
        .into_iter()
        .map(|item| {
            SearchResult::new(
                item.title.unwrap_or_default(),
                item.url.unwrap_or_default(),
                item.description.unwrap_or_default(),
            )
        })
        .collect();

    Ok(results)
}

/// Convert a `ureq` error into our own [`WebSearchError`], including the HTTP
/// body for non-success statuses so failures are easy to diagnose.
fn map_ureq_error(err: ureq::Error) -> WebSearchError {
    match err {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            WebSearchError::Http(format!("HTTP {code}: {body}"))
        }
        ureq::Error::Transport(transport) => WebSearchError::Http(transport.to_string()),
    }
}

/// The subset of the Brave Web Search response we care about.
///
/// Brave returns many sibling blocks (`news`, `videos`, `infobox`, ...); we only
/// read `web`, and `serde` ignores the rest.
#[derive(Debug, Deserialize)]
struct BraveResponse {
    /// The web-results block. Absent when Brave returns no web results.
    web: Option<BraveWeb>,
}

/// The `web` block of a Brave response.
#[derive(Debug, Deserialize)]
struct BraveWeb {
    /// The ranked list of web results. Defaults to empty when omitted.
    #[serde(default)]
    results: Vec<BraveResult>,
}

/// One entry in `web.results[]`. All fields are optional so a partial result
/// never fails the whole decode.
#[derive(Debug, Deserialize)]
struct BraveResult {
    title: Option<String>,
    url: Option<String>,
    description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_encode_leaves_unreserved_characters_untouched() {
        assert_eq!(percent_encode("Gaia-Robot_1.0~ok"), "Gaia-Robot_1.0~ok");
    }

    #[test]
    fn percent_encode_escapes_spaces_and_reserved_characters() {
        // Space -> %20, & -> %26, = -> %3D.
        assert_eq!(percent_encode("a b&c=d"), "a%20b%26c%3Dd");
    }

    #[test]
    fn percent_encode_escapes_non_ascii_utf8_bytes() {
        // 'é' is U+00E9 -> UTF-8 0xC3 0xA9.
        assert_eq!(percent_encode("é"), "%C3%A9");
    }

    #[test]
    fn clamp_count_uses_default_for_zero() {
        assert_eq!(clamp_count(0), DEFAULT_RESULT_COUNT);
    }

    #[test]
    fn clamp_count_caps_at_the_maximum() {
        assert_eq!(clamp_count(1000), MAX_RESULT_COUNT);
        assert_eq!(clamp_count(3), 3);
    }

    #[test]
    fn build_search_url_encodes_query_and_includes_count() {
        let url = build_search_url(DEFAULT_BRAVE_ENDPOINT, "time in Johannesburg", 3);
        assert_eq!(
            url,
            "https://api.search.brave.com/res/v1/web/search?q=time%20in%20Johannesburg&count=3"
        );
    }

    #[test]
    fn build_search_url_honours_a_custom_endpoint() {
        let url = build_search_url("https://example.test/search", "rust", 5);
        assert_eq!(url, "https://example.test/search?q=rust&count=5");
    }

    #[test]
    fn parse_results_maps_web_results_in_order() {
        let body = r#"{
            "web": {
                "results": [
                    { "title": "First", "url": "https://example.com/1", "description": "one" },
                    { "title": "Second", "url": "https://example.com/2", "description": "two" }
                ]
            }
        }"#;

        let results = parse_results(body).expect("valid body should parse");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "First");
        assert_eq!(results[0].url, "https://example.com/1");
        assert_eq!(results[0].snippet, "one");
        assert_eq!(results[1].title, "Second");
    }

    #[test]
    fn parse_results_defaults_missing_fields_to_empty_strings() {
        let body = r#"{ "web": { "results": [ { "url": "https://example.com/x" } ] } }"#;

        let results = parse_results(body).expect("partial result should still parse");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "");
        assert_eq!(results[0].url, "https://example.com/x");
        assert_eq!(results[0].snippet, "");
    }

    #[test]
    fn parse_results_returns_empty_when_there_is_no_web_block() {
        // A valid response that simply carried no web results.
        let results = parse_results(r#"{ "query": { "original": "x" } }"#)
            .expect("a response without a web block is still valid");
        assert!(results.is_empty());
    }

    #[test]
    fn parse_results_errors_on_invalid_json() {
        let err = parse_results("not json").expect_err("garbage should not decode");
        assert!(matches!(err, WebSearchError::Decode(_)));
    }

    #[test]
    fn from_parts_returns_none_without_an_api_key() {
        // No key configured -> no client, regardless of the endpoint.
        assert!(from_parts(None, None).is_none());
        assert!(from_parts(None, Some("https://example.test/search".to_string())).is_none());
    }

    #[test]
    fn from_parts_uses_the_default_endpoint_when_unset() {
        let client = from_parts(Some("secret-token".to_string()), None)
            .expect("a key with no endpoint should still build a client");
        assert_eq!(client.endpoint(), DEFAULT_BRAVE_ENDPOINT);
    }

    #[test]
    fn from_parts_honours_a_custom_endpoint() {
        let client = from_parts(
            Some("secret-token".to_string()),
            Some("https://example.test/search".to_string()),
        )
        .expect("a key with an endpoint should build a client");
        assert_eq!(client.endpoint(), "https://example.test/search");
    }

    #[test]
    fn display_renders_http_and_decode_variants() {
        let http = WebSearchError::Http("HTTP 500: boom".to_string());
        assert_eq!(
            http.to_string(),
            "web search request failed: HTTP 500: boom"
        );
        let decode = WebSearchError::Decode("bad json".to_string());
        assert_eq!(
            decode.to_string(),
            "could not decode web search response: bad json"
        );
    }

    #[test]
    fn search_sends_the_request_and_maps_the_results() {
        // A 200 with a Brave-shaped body exercises the real GET + parse path.
        let body = r#"{"web":{"results":[
            {"title":"Mars","url":"https://example.com/mars","description":"red planet"}
        ]}}"#;
        let (endpoint, handle) = crate::test_http::spawn_mock_http("200 OK", body);
        let client = from_parts(Some("token".to_string()), Some(endpoint))
            .expect("client builds with a key and endpoint");

        let results = client
            .search("mars", 3)
            .expect("search succeeds against the mock");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Mars");
        assert_eq!(results[0].url, "https://example.com/mars");
        assert_eq!(results[0].snippet, "red planet");
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn search_maps_a_non_success_status_into_an_http_error() {
        let (endpoint, handle) = crate::test_http::spawn_mock_http(
            "429 Too Many Requests",
            r#"{"message":"slow down"}"#,
        );
        let client = from_parts(Some("token".to_string()), Some(endpoint))
            .expect("client builds with a key and endpoint");

        let err = client
            .search("mars", 3)
            .expect_err("a 429 must surface as an error");
        match err {
            WebSearchError::Http(msg) => assert!(msg.contains("429"), "got: {msg}"),
            other => panic!("expected an Http error, got {other:?}"),
        }
        handle.join().expect("mock server thread joins");
    }
}
