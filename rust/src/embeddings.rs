//! The [`EmbeddingClient`] type: query-time embeddings for Cosmos vector search.
//!
//! Gaia's Cosmos containers already store per-record embeddings at `/dataVector`
//! and index them with DiskANN. To execute a semantic query at runtime we still
//! need one missing piece: embed the query text itself, then bind that vector as
//! a query parameter for `VectorDistance(...)`.
//!
//! This module is that piece. It calls the Azure OpenAI embeddings endpoint for
//! the configured Foundry deployment and returns a single `Vec<f32>` per input.
//!
//! Configuration is read from the environment (falling back to `infra/.env` via
//! [`crate::llm::value_from_env`]):
//! - `FOUNDRY_ENDPOINT` (required when semantic search is enabled)
//! - `EMBEDDING_DEPLOYMENT` (required when semantic search is enabled)
//! - `EMBEDDING_DIMENSIONS` or `COSMOS_VECTOR_DIMS` (optional)
//! - `AZURE_OPENAI_API_VERSION` or `FOUNDRY_API_VERSION` (optional)
//! - `FOUNDRY_API_KEY` (preferred credential) or `FOUNDRY_AAD_TOKEN` (fallback)
//! - managed identity token (auto-fallback when running in Azure and no env
//!   key/token is provided)
//!
//! Authentication mirrors `llm.rs`:
//! - API key -> `api-key: <key>`
//! - AAD token -> `Authorization: Bearer <token>`

use std::fmt;

use serde::{Deserialize, Serialize};

const DEFAULT_API_VERSION: &str = "2024-10-21";

/// Errors that can occur while configuring or calling embeddings.
#[derive(Debug)]
pub enum EmbeddingError {
    /// Endpoint/deployment are configured but no credential was found.
    MissingCredential,
    /// The configured dimensions value was not a valid integer.
    InvalidDimensions(String),
    /// The HTTP request failed or returned a non-success status.
    Http(String),
    /// The response body could not be decoded into the expected shape.
    Decode(String),
    /// The service returned no embedding payload.
    EmptyResponse,
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingError::MissingCredential => write!(
                f,
                "no Foundry embedding credential found (set FOUNDRY_API_KEY or FOUNDRY_AAD_TOKEN)"
            ),
            EmbeddingError::InvalidDimensions(raw) => {
                write!(
                    f,
                    "invalid EMBEDDING_DIMENSIONS/COSMOS_VECTOR_DIMS value: {raw}"
                )
            }
            EmbeddingError::Http(msg) => write!(f, "embedding request failed: {msg}"),
            EmbeddingError::Decode(msg) => write!(f, "could not decode embedding response: {msg}"),
            EmbeddingError::EmptyResponse => write!(f, "embedding response was empty"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// How this client authenticates requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthScheme {
    ApiKey,
    Bearer,
}

/// A minimal, immutable client for one embedding deployment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingClient {
    endpoint: String,
    auth_token: String,
    auth_scheme: AuthScheme,
    dimensions: Option<u32>,
}

impl EmbeddingClient {
    /// Build a client from environment values, or `Ok(None)` when embeddings are
    /// not configured for this run.
    ///
    /// Both `FOUNDRY_ENDPOINT` and `EMBEDDING_DEPLOYMENT` must be present to
    /// enable the client. This keeps semantic search opt-in and preserves the
    /// existing keyword-only behavior in environments that have not configured
    /// embedding infrastructure yet.
    pub fn from_env() -> Result<Option<Self>, EmbeddingError> {
        let endpoint = crate::llm::value_from_env("FOUNDRY_ENDPOINT");
        let deployment = crate::llm::value_from_env("EMBEDDING_DEPLOYMENT");

        let (endpoint, deployment) = match (endpoint, deployment) {
            (Some(endpoint), Some(deployment))
                if !endpoint.trim().is_empty() && !deployment.trim().is_empty() =>
            {
                (endpoint, deployment)
            }
            _ => return Ok(None),
        };

        let api_version = crate::llm::value_from_env("AZURE_OPENAI_API_VERSION")
            .or_else(|| crate::llm::value_from_env("FOUNDRY_API_VERSION"))
            .unwrap_or_else(|| DEFAULT_API_VERSION.to_string());

        let endpoint = build_embeddings_url(&endpoint, &deployment, &api_version);

        let dimensions = match crate::llm::value_from_env("EMBEDDING_DIMENSIONS")
            .or_else(|| crate::llm::value_from_env("COSMOS_VECTOR_DIMS"))
        {
            Some(raw) if !raw.trim().is_empty() => {
                let parsed = raw
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| EmbeddingError::InvalidDimensions(raw.clone()))?;
                if parsed == 0 {
                    None
                } else {
                    Some(parsed)
                }
            }
            _ => None,
        };

        let api_key =
            crate::llm::value_from_env("FOUNDRY_API_KEY").filter(|value| !value.trim().is_empty());
        let aad = crate::llm::value_from_env("FOUNDRY_AAD_TOKEN")
            .filter(|value| !value.trim().is_empty());

        let (auth_token, auth_scheme) = match (api_key, aad) {
            (Some(token), _) => (token, AuthScheme::ApiKey),
            (None, Some(token)) => (token, AuthScheme::Bearer),
            (None, None) => {
                if let Some(token) =
                    crate::llm::managed_identity_token("https://cognitiveservices.azure.com")
                {
                    (token, AuthScheme::Bearer)
                } else {
                    return Err(EmbeddingError::MissingCredential);
                }
            }
        };

        Ok(Some(Self {
            endpoint,
            auth_token,
            auth_scheme,
            dimensions,
        }))
    }

    /// The embeddings endpoint this client posts to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Test-only constructor pointing the client at `endpoint` with an API-key
    /// credential, so other modules' tests (notably the write controller) can
    /// drive [`embed`](EmbeddingClient::embed) against a mock server.
    #[cfg(test)]
    pub(crate) fn for_test(endpoint: String) -> Self {
        Self {
            endpoint,
            auth_token: "secret-key".to_string(),
            auth_scheme: AuthScheme::ApiKey,
            dimensions: None,
        }
    }

    /// Probe Foundry connectivity and RBAC by embedding a tiny string.
    ///
    /// Runs a real embedding of `"ping"`, which exercises the full production
    /// path: endpoint reachability, the auth scheme (managed-identity bearer or
    /// Foundry API key), the **Cognitive Services OpenAI User** permission, and
    /// the embedding deployment itself. This is the same Foundry account and
    /// token the model-router uses, so a success also confirms that account's
    /// RBAC. The vector is discarded; `Ok(())` means the call succeeded.
    pub fn ping(&self) -> Result<(), EmbeddingError> {
        self.embed("ping").map(|_| ())
    }

    /// Embed one piece of text into a query vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let cleaned = if text.trim().is_empty() {
            " "
        } else {
            text.trim()
        };
        let body = EmbeddingRequest {
            input: cleaned,
            dimensions: self.dimensions,
        };
        let payload =
            serde_json::to_vec(&body).map_err(|e| EmbeddingError::Decode(e.to_string()))?;

        let request = ureq::post(&self.endpoint).set("Content-Type", "application/json");
        let request = match self.auth_scheme {
            AuthScheme::ApiKey => request.set("api-key", &self.auth_token),
            AuthScheme::Bearer => {
                request.set("Authorization", &format!("Bearer {}", self.auth_token))
            }
        };

        let response = request.send_bytes(&payload).map_err(map_ureq_error)?;
        let raw = response
            .into_string()
            .map_err(|e| EmbeddingError::Http(e.to_string()))?;

        parse_embedding(&raw)
    }
}

/// Build an Azure OpenAI embeddings URL for one deployment.
fn build_embeddings_url(endpoint: &str, deployment: &str, api_version: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    format!("{base}/openai/deployments/{deployment}/embeddings?api-version={api_version}")
}

/// Parse the first embedding payload from an Azure OpenAI response body.
fn parse_embedding(body: &str) -> Result<Vec<f32>, EmbeddingError> {
    let parsed: EmbeddingResponse =
        serde_json::from_str(body).map_err(|e| EmbeddingError::Decode(e.to_string()))?;

    parsed
        .data
        .into_iter()
        .min_by_key(|item| item.index)
        .map(|item| item.embedding)
        .ok_or(EmbeddingError::EmptyResponse)
}

fn map_ureq_error(err: ureq::Error) -> EmbeddingError {
    match err {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            EmbeddingError::Http(format!("HTTP {code}: {body}"))
        }
        ureq::Error::Transport(transport) => EmbeddingError::Http(transport.to_string()),
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    #[serde(default)]
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_embeddings_url_from_endpoint_deployment_and_version() {
        let url = build_embeddings_url(
            "https://gaia-foundry.openai.azure.com/",
            "text-embedding",
            "2024-10-21",
        );
        assert_eq!(
            url,
            "https://gaia-foundry.openai.azure.com/openai/deployments/text-embedding/embeddings?api-version=2024-10-21"
        );
    }

    #[test]
    fn parse_embedding_returns_the_first_index_entry() {
        let body = r#"{
            "data": [
                {"index": 1, "embedding": [9.0, 9.0]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]}
            ]
        }"#;

        let vector = parse_embedding(body).unwrap();
        assert_eq!(vector, vec![0.1_f32, 0.2_f32, 0.3_f32]);
    }

    #[test]
    fn parse_embedding_rejects_empty_payloads() {
        let body = r#"{"data":[]}"#;
        let err = parse_embedding(body).unwrap_err();
        assert!(matches!(err, EmbeddingError::EmptyResponse));
    }

    #[test]
    fn display_renders_each_error_variant() {
        assert!(EmbeddingError::MissingCredential
            .to_string()
            .contains("no Foundry embedding credential"));
        assert!(EmbeddingError::InvalidDimensions("nope".to_string())
            .to_string()
            .contains("nope"));
        assert!(EmbeddingError::Http("boom".to_string())
            .to_string()
            .contains("boom"));
        assert!(EmbeddingError::Decode("bad".to_string())
            .to_string()
            .contains("bad"));
        assert!(EmbeddingError::EmptyResponse
            .to_string()
            .contains("response was empty"));
    }

    /// Build a client that posts to `endpoint` with an API-key credential.
    fn client_for(endpoint: String) -> EmbeddingClient {
        EmbeddingClient {
            endpoint,
            auth_token: "secret-key".to_string(),
            auth_scheme: AuthScheme::ApiKey,
            dimensions: Some(3),
        }
    }

    #[test]
    fn embed_posts_the_text_and_returns_the_query_vector() {
        let body = r#"{"data":[{"index":0,"embedding":[0.5,0.25,0.125]}]}"#;
        let (endpoint, handle) = crate::test_http::spawn_mock_http("200 OK", body);

        let vector = client_for(endpoint)
            .embed("hello world")
            .expect("embed succeeds against the mock");

        assert_eq!(vector, vec![0.5_f32, 0.25_f32, 0.125_f32]);
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn embed_blank_text_still_sends_a_request() {
        // Whitespace-only input is normalised to a single space, not rejected.
        let body = r#"{"data":[{"index":0,"embedding":[1.0]}]}"#;
        let (endpoint, handle) = crate::test_http::spawn_mock_http("200 OK", body);

        let vector = client_for(endpoint)
            .embed("   ")
            .expect("a blank query is embedded, not skipped");

        assert_eq!(vector, vec![1.0_f32]);
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn embed_maps_a_non_success_status_into_an_http_error() {
        let (endpoint, handle) =
            crate::test_http::spawn_mock_http("401 Unauthorized", r#"{"error":"bad key"}"#);

        let err = client_for(endpoint)
            .embed("hello")
            .expect_err("a 401 must surface as an error");
        match err {
            EmbeddingError::Http(msg) => assert!(msg.contains("401"), "got: {msg}"),
            other => panic!("expected an Http error, got {other:?}"),
        }
        handle.join().expect("mock server thread joins");
    }
}
