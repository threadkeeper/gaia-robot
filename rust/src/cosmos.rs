#![allow(dead_code)]

//! The [`CosmosClient`] type: a tiny, dependency-light client for Gaia's Azure
//! Cosmos DB for NoSQL containers (`GaiaKB`, `GaiaDiary`, `UsersKB`,
//! `UsersDataLake`, `GaiaDataLake`).
//!
//! Like [`crate::llm::LlmClient`], this client is **opt-in** and reuses the same
//! dev/local switch (`GAIA_MODE=dev`/`local`). When it is not configured,
//! [`CosmosClient::from_env`] returns `Ok(None)` and the program keeps its
//! offline skeleton behaviour.
//!
//! ## Why a hand-rolled REST client?
//!
//! There is no first-party Azure Cosmos SDK on crates.io, and the official
//! options pull in large async runtimes. To honour this repo's
//! minimise-dependencies rule we talk to the Cosmos REST API directly over
//! [`ureq`] (already a dependency) using **Azure AD bearer-token** auth:
//!
//! - The caller supplies an AAD access token via `COSMOS_AAD_TOKEN`, e.g.
//!   `az account get-access-token --resource https://<account>.documents.azure.com`.
//! - That avoids the HMAC-SHA256 master-key signing scheme, so we need **no**
//!   crypto crates. (Data-plane item read/write works with an AAD token that has
//!   the *Cosmos DB Built-in Data Contributor* role — see `infra` notes.)
//!
//! All header/URL/parsing logic lives in small pure functions so it can be unit
//! tested without a network or a live Cosmos account.
//!
//! Configuration (environment, falling back to `infra/.env`):
//! - `COSMOS_ENDPOINT` — account URL, e.g. `https://acct.documents.azure.com:443/`.
//! - `COSMOS_AAD_TOKEN` — AAD access token for the Cosmos data plane.
//! - `COSMOS_DATABASE` — database name (default `gaia`).

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::storage::Record;

/// REST API version sent in the `x-ms-version` header. `2018-12-31` covers the
/// point reads, upserts, and single-partition parameterised queries we use.
const API_VERSION: &str = "2018-12-31";

/// Hard cap on the number of documents a single query response may return.
///
/// Sent as `x-ms-max-item-count`. Queries should already bound themselves with
/// `TOP n`, but this guarantees that even a query that forgot its limit can
/// never flood the in-memory Response Data Context handed to LLM Call 2. We read
/// a single response page (no continuation), so this is an absolute ceiling.
const MAX_ITEMS_PER_QUERY: &str = "100";

/// Errors that can occur while configuring or calling Cosmos DB.
#[derive(Debug)]
pub enum CosmosError {
    /// Cosmos was requested but no AAD token could be found.
    MissingToken,
    /// A managed-identity (SAMI) token could not be acquired from the host.
    Token(String),
    /// The HTTP request failed or returned a non-success status.
    Http(String),
    /// A response body could not be decoded into the expected shape.
    Decode(String),
}

impl fmt::Display for CosmosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CosmosError::MissingToken => write!(
                f,
                "no Cosmos AAD token found (set COSMOS_AAD_TOKEN or put it in infra/.env)"
            ),
            CosmosError::Token(msg) => write!(f, "Cosmos managed-identity token error: {msg}"),
            CosmosError::Http(msg) => write!(f, "Cosmos request failed: {msg}"),
            CosmosError::Decode(msg) => write!(f, "could not decode Cosmos response: {msg}"),
        }
    }
}

impl std::error::Error for CosmosError {}

/// A named query parameter, mirroring Cosmos's `{"name": "@p", "value": ...}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryParam {
    /// Parameter name including the leading `@`, e.g. `@userId`.
    pub name: String,
    /// Parameter value; any JSON scalar Cosmos accepts.
    pub value: serde_json::Value,
}

impl QueryParam {
    /// Create a query parameter from a name and any JSON-convertible value.
    pub fn new(name: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// How long a freshly minted managed-identity token is trusted before it is
/// re-minted. AAD access tokens live ~60-90 minutes; refreshing well ahead of
/// that (45 minutes) guarantees an in-flight request never uses a dead token.
const MANAGED_IDENTITY_TTL_SECS: u64 = 45 * 60;

/// A managed-identity token plus the instant after which it must be re-minted.
#[derive(Clone)]
struct CachedToken {
    /// The bearer token value.
    value: String,
    /// Unix seconds after which [`Credential::bearer`] re-mints the token.
    refresh_after: u64,
}

/// How a [`CosmosClient`] authenticates to the Cosmos DB data plane.
///
/// Both variants ultimately produce an Azure AD bearer token; they differ only
/// in where that token comes from and whether it is refreshed.
#[derive(Clone)]
enum Credential {
    /// A pre-minted AAD access token supplied via `COSMOS_AAD_TOKEN`.
    ///
    /// Convenient for local development (where `infra/run-local.ps1` mints a
    /// short-lived token with the Azure CLI), but it expires after ~1 hour and
    /// is never refreshed, so it is unsuitable for long-running deployments.
    Static(String),
    /// The host's **system-assigned managed identity** (SAMI) — the production
    /// path. Tokens are minted on demand from the local identity endpoint and
    /// cached until shortly before they expire, then transparently refreshed.
    /// No secret or static token is needed: the Container App's identity is
    /// granted the Cosmos *Built-in Data Contributor* role at deploy time.
    ManagedIdentity {
        /// AAD resource/audience to request, e.g.
        /// `https://<account>.documents.azure.com`.
        resource: String,
        /// Token cache shared across clones so a single refresh serves them all.
        cache: Arc<Mutex<Option<CachedToken>>>,
    },
}

impl Credential {
    /// A short, secret-free label for diagnostics (`"static"` / `"managed-identity"`).
    fn kind(&self) -> &'static str {
        match self {
            Credential::Static(_) => "static",
            Credential::ManagedIdentity { .. } => "managed-identity",
        }
    }

    /// Resolve the bearer token to send on the next request.
    ///
    /// Static credentials return their token verbatim. Managed-identity
    /// credentials return a cached token while it is still fresh and otherwise
    /// mint a new one from the local identity endpoint, caching it for reuse.
    fn bearer(&self) -> Result<String, CosmosError> {
        match self {
            Credential::Static(token) => Ok(token.clone()),
            Credential::ManagedIdentity { resource, cache } => {
                let now = unix_now();
                // Fast path: a cached token that is still comfortably valid.
                if let Ok(guard) = cache.lock() {
                    if let Some(cached) = guard.as_ref() {
                        if now < cached.refresh_after {
                            return Ok(cached.value.clone());
                        }
                    }
                }
                // Mint a fresh token from the host's managed-identity endpoint.
                let token = crate::llm::managed_identity_token(resource).ok_or_else(|| {
                    CosmosError::Token(format!(
                        "could not obtain a managed-identity token for {resource} \
                         (is the host's system-assigned identity enabled and granted \
                         the Cosmos Built-in Data Contributor role?)"
                    ))
                })?;
                if let Ok(mut guard) = cache.lock() {
                    *guard = Some(CachedToken {
                        value: token.clone(),
                        refresh_after: now + MANAGED_IDENTITY_TTL_SECS,
                    });
                }
                Ok(token)
            }
        }
    }
}

/// A minimal, immutable client bound to one Cosmos account + database.
///
/// Construct with [`CosmosClient::from_env`] and call [`CosmosClient::query`],
/// [`CosmosClient::upsert`], or [`CosmosClient::get`]. Cheap to clone; managed
/// identity clones share one token cache so refreshes are not duplicated.
#[derive(Clone)]
pub struct CosmosClient {
    /// Account URL, always normalised to end with a single `/`.
    endpoint: String,
    /// Database name, e.g. `gaia`.
    database: String,
    /// How the client authenticates to the Cosmos data plane.
    cred: Credential,
}

// Hand-written so the bearer token (a secret) is never printed in logs or
// panic messages; only the endpoint, database, and auth *kind* are shown.
impl fmt::Debug for CosmosClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CosmosClient")
            .field("endpoint", &self.endpoint)
            .field("database", &self.database)
            .field("auth", &self.cred.kind())
            .finish()
    }
}

impl CosmosClient {
    /// Build a client from a pre-minted static AAD token (used by tests and the
    /// `COSMOS_AAD_TOKEN` dev path).
    ///
    /// The endpoint is normalised to end with exactly one `/` so URL building is
    /// simple and predictable.
    ///
    /// [`from_env`]: CosmosClient::from_env
    pub fn new(
        endpoint: impl Into<String>,
        database: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: normalise_endpoint(endpoint.into()),
            database: database.into(),
            cred: Credential::Static(token.into()),
        }
    }

    /// Build a client that authenticates with the host's system-assigned
    /// managed identity (SAMI), minting and refreshing Cosmos data-plane tokens
    /// on demand. `resource` is the AAD audience, e.g.
    /// `https://<account>.documents.azure.com`.
    pub fn with_managed_identity(
        endpoint: impl Into<String>,
        database: impl Into<String>,
        resource: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: normalise_endpoint(endpoint.into()),
            database: database.into(),
            cred: Credential::ManagedIdentity {
                resource: resource.into(),
                cache: Arc::new(Mutex::new(None)),
            },
        }
    }

    /// The bearer token to authenticate the next request, refreshing a
    /// managed-identity token when the cached one is stale.
    fn bearer(&self) -> Result<String, CosmosError> {
        self.cred.bearer()
    }

    /// Build a client from the environment, or `Ok(None)` when Cosmos is not in
    /// use for this run.
    ///
    /// Returns:
    /// - `Ok(None)` when dev/local mode is off, or `COSMOS_ENDPOINT` is unset
    ///   (the program then stays fully offline).
    /// - `Ok(Some(client))` otherwise. Authentication is chosen automatically:
    ///   a `COSMOS_AAD_TOKEN`, when present, is used as a static token (the
    ///   local dev convenience); when absent, the client falls back to the
    ///   host's **system-assigned managed identity** (the production path).
    ///   Because managed identity needs no secret, this no longer fails with a
    ///   missing-token error; any auth problem instead surfaces at write time
    ///   as a visible Cosmos error.
    pub fn from_env() -> Result<Option<Self>, CosmosError> {
        if !dev_mode_enabled() {
            return Ok(None);
        }

        let endpoint = match resolve_env("COSMOS_ENDPOINT") {
            Some(value) if !value.is_empty() => value,
            // No endpoint configured: Cosmos simply isn't part of this run.
            _ => return Ok(None),
        };

        let database = resolve_env("COSMOS_DATABASE")
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "gaia".to_string());

        match resolve_env("COSMOS_AAD_TOKEN").filter(|token| !token.is_empty()) {
            // A pre-minted token was supplied (local dev): use it directly.
            Some(token) => Ok(Some(Self::new(endpoint, database, token))),
            // No token: authenticate with the host's managed identity (SAMI).
            None => {
                let resource = cosmos_token_resource(&endpoint);
                Ok(Some(Self::with_managed_identity(
                    endpoint, database, resource,
                )))
            }
        }
    }

    /// The database this client targets.
    pub fn database(&self) -> &str {
        &self.database
    }

    /// The (normalised) account endpoint this client targets.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Run a parameterised SQL query against a single logical partition.
    ///
    /// `partition_value` is the entity/userId that scopes the query to one
    /// partition (cheap and avoids a cross-partition fan-out). The returned
    /// documents are deserialized into [`Record`]s.
    pub fn query(
        &self,
        container: &str,
        partition_value: &str,
        query: &str,
        params: &[QueryParam],
    ) -> Result<Vec<Record>, CosmosError> {
        let url = docs_url(&self.endpoint, &self.database, container);
        let body = serde_json::to_vec(&QueryBody {
            query,
            parameters: params,
        })
        .map_err(|e| CosmosError::Decode(e.to_string()))?;

        let auth = aad_auth_header(&self.bearer()?);
        let response = ureq::post(&url)
            .set("Authorization", &auth)
            .set("x-ms-date", &now_rfc1123())
            .set("x-ms-version", API_VERSION)
            .set("x-ms-documentdb-isquery", "true")
            .set("Content-Type", "application/query+json")
            .set(
                "x-ms-documentdb-partitionkey",
                &partition_key_header(partition_value),
            )
            .set("x-ms-documentdb-query-enablecrosspartition", "false")
            .set("x-ms-max-item-count", MAX_ITEMS_PER_QUERY)
            .send_bytes(&body)
            .map_err(map_ureq_error)?;

        let text = response
            .into_string()
            .map_err(|e| CosmosError::Http(e.to_string()))?;

        parse_documents(&text)
    }

    /// Upsert (insert-or-replace) one [`Record`] into a container.
    ///
    /// `partition_value` must equal the record's business key (entity/userId);
    /// Cosmos enforces that the document's partition field matches this header.
    pub fn upsert(
        &self,
        container: &str,
        partition_value: &str,
        record: &Record,
    ) -> Result<(), CosmosError> {
        // A [`Record`] is just one serializable document shape; delegate to the
        // generic path so there is a single upsert code path on the wire.
        self.upsert_doc(container, partition_value, record)
    }

    /// Upsert (insert-or-replace) any serializable document into a container.
    ///
    /// This is the generic counterpart to [`upsert`](CosmosClient::upsert): it
    /// accepts any `Serialize` type, so callers can write documents whose shape
    /// differs from [`Record`] (for example the `DataLakeIndex` entries, which
    /// carry an `indexVector` and `source` rather than `dataVector`).
    ///
    /// `partition_value` must equal the document's partition field; Cosmos
    /// enforces that the two match.
    pub fn upsert_doc<T: Serialize>(
        &self,
        container: &str,
        partition_value: &str,
        doc: &T,
    ) -> Result<(), CosmosError> {
        let url = docs_url(&self.endpoint, &self.database, container);
        let body = serde_json::to_vec(doc).map_err(|e| CosmosError::Decode(e.to_string()))?;

        let auth = aad_auth_header(&self.bearer()?);
        ureq::post(&url)
            .set("Authorization", &auth)
            .set("x-ms-date", &now_rfc1123())
            .set("x-ms-version", API_VERSION)
            .set("x-ms-documentdb-is-upsert", "true")
            .set(
                "x-ms-documentdb-partitionkey",
                &partition_key_header(partition_value),
            )
            .set("Content-Type", "application/json")
            .send_bytes(&body)
            .map_err(map_ureq_error)?;

        Ok(())
    }

    /// Point-read a single document by id within a partition.
    ///
    /// Returns `Ok(None)` when the document does not exist (HTTP 404) rather than
    /// treating a miss as an error.
    pub fn get(
        &self,
        container: &str,
        partition_value: &str,
        id: &str,
    ) -> Result<Option<Record>, CosmosError> {
        let url = format!(
            "{}{}",
            docs_url(&self.endpoint, &self.database, container),
            format_args!("/{id}"),
        );

        let auth = aad_auth_header(&self.bearer()?);
        let result = ureq::get(&url)
            .set("Authorization", &auth)
            .set("x-ms-date", &now_rfc1123())
            .set("x-ms-version", API_VERSION)
            .set(
                "x-ms-documentdb-partitionkey",
                &partition_key_header(partition_value),
            )
            .call();

        match result {
            Ok(response) => {
                let text = response
                    .into_string()
                    .map_err(|e| CosmosError::Http(e.to_string()))?;
                let record =
                    serde_json::from_str(&text).map_err(|e| CosmosError::Decode(e.to_string()))?;
                Ok(Some(record))
            }
            // A missing document is a normal "not found", not a failure.
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(err) => Err(map_ureq_error(err)),
        }
    }

    /// Point-read a single document by id as a raw JSON [`serde_json::Value`].
    ///
    /// The generic counterpart to [`get`](CosmosClient::get): use it for
    /// documents whose shape is not a [`Record`] (for example `DataLakeIndex`
    /// entries, which carry an `indexVector`). Returns `Ok(None)` on HTTP 404.
    pub fn get_value(
        &self,
        container: &str,
        partition_value: &str,
        id: &str,
    ) -> Result<Option<serde_json::Value>, CosmosError> {
        let url = format!(
            "{}{}",
            docs_url(&self.endpoint, &self.database, container),
            format_args!("/{id}"),
        );

        let auth = aad_auth_header(&self.bearer()?);
        let result = ureq::get(&url)
            .set("Authorization", &auth)
            .set("x-ms-date", &now_rfc1123())
            .set("x-ms-version", API_VERSION)
            .set(
                "x-ms-documentdb-partitionkey",
                &partition_key_header(partition_value),
            )
            .call();

        match result {
            Ok(response) => {
                let text = response
                    .into_string()
                    .map_err(|e| CosmosError::Http(e.to_string()))?;
                let value =
                    serde_json::from_str(&text).map_err(|e| CosmosError::Decode(e.to_string()))?;
                Ok(Some(value))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(err) => Err(map_ureq_error(err)),
        }
    }
}

// --- Pure helpers (no network) ----------------------------------------------

/// The Cosmos `docs` collection URL for one container.
fn docs_url(endpoint: &str, database: &str, container: &str) -> String {
    // `endpoint` is guaranteed to end with '/' by `CosmosClient::new`.
    format!("{endpoint}dbs/{database}/colls/{container}/docs")
}

/// Normalise a Cosmos account URL to end with exactly one `/`.
///
/// Centralised so every constructor produces identical, predictable endpoints
/// for URL building.
fn normalise_endpoint(mut endpoint: String) -> String {
    if !endpoint.ends_with('/') {
        endpoint.push('/');
    }
    endpoint
}

/// Current time as whole seconds since the Unix epoch.
///
/// A clock skewed before 1970 is treated as `0`; the only consumer is the
/// managed-identity token TTL, where a slightly early refresh is harmless.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// Derive the AAD resource/audience for a Cosmos data-plane token from an
/// account endpoint.
///
/// The Cosmos token audience is the bare `scheme://host` of the account, with
/// no port, path, query, or trailing slash — e.g.
/// `https://acct.documents.azure.com:443/` becomes
/// `https://acct.documents.azure.com`.
fn cosmos_token_resource(endpoint: &str) -> String {
    // Split off the scheme (default to https if somehow absent).
    let (scheme, rest) = match endpoint.split_once("://") {
        Some((scheme, rest)) => (scheme, rest),
        None => ("https", endpoint),
    };
    // The host is everything up to the first `/`, `:` (port), or `?` (query).
    let host_end = rest.find(['/', ':', '?']).unwrap_or(rest.len());
    let host = &rest[..host_end];
    format!("{scheme}://{host}")
}

/// Build the `Authorization` header for Azure AD bearer-token auth.
///
/// Cosmos expects the URL-encoded string `type=aad&ver=1.0&sig=<token>`. AAD
/// access tokens are URL-safe (base64url segments joined by `.`), so only the
/// fixed prefix needs percent-encoding — which we can hard-code.
fn aad_auth_header(token: &str) -> String {
    format!("type%3Daad%26ver%3D1.0%26sig%3D{token}")
}

/// Build the `x-ms-documentdb-partitionkey` header value, e.g. `["user-1"]`.
///
/// Serialising through `serde_json` correctly escapes any quotes/backslashes in
/// the partition value.
fn partition_key_header(value: &str) -> String {
    // A single-element JSON array of the partition value.
    serde_json::Value::Array(vec![serde_json::Value::String(value.to_string())]).to_string()
}

/// Parse a Cosmos query response body (`{"Documents": [...]}`) into records.
fn parse_documents(body: &str) -> Result<Vec<Record>, CosmosError> {
    let parsed: DocumentsResponse =
        serde_json::from_str(body).map_err(|e| CosmosError::Decode(e.to_string()))?;
    Ok(parsed.documents)
}

/// Current time as an RFC 1123 GMT string for the `x-ms-date` header.
fn now_rfc1123() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    rfc1123(secs)
}

/// Format seconds-since-Unix-epoch as `Tue, 16 Jun 2026 12:00:00 GMT`.
///
/// Implemented with std only (no `chrono`/`time`) to avoid a dependency. Uses
/// Howard Hinnant's civil-from-days algorithm for the calendar date.
fn rfc1123(secs_since_epoch: u64) -> String {
    const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    let days = (secs_since_epoch / 86_400) as i64;
    let secs_of_day = secs_since_epoch % 86_400;
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;

    // 1970-01-01 (day 0) was a Thursday; 0 = Sunday. `days` is always >= 0.
    let weekday = ((days % 7) + 4) % 7;
    let (year, month, day) = civil_from_days(days);

    format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        WEEKDAYS[weekday as usize],
        day,
        MONTHS[(month - 1) as usize],
        year,
        hour,
        minute,
        second,
    )
}

/// Convert a day count since 1970-01-01 into a `(year, month, day)` triple.
///
/// This is Howard Hinnant's well-known `civil_from_days` algorithm; the magic
/// constants come from the proleptic Gregorian calendar's 400-year cycle.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year, [0, 365]
    let mp = (5 * doy + 2) / 153; // month index, [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// Convert a `ureq` error into a [`CosmosError`], keeping the HTTP body so
/// failures (auth, missing container, RBAC) are easy to diagnose.
fn map_ureq_error(err: ureq::Error) -> CosmosError {
    match err {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            CosmosError::Http(format!("HTTP {code}: {body}"))
        }
        ureq::Error::Transport(transport) => CosmosError::Http(transport.to_string()),
    }
}

/// Whether dev/local mode is enabled via `GAIA_MODE` (shared with the LLM path).
fn dev_mode_enabled() -> bool {
    match std::env::var("GAIA_MODE") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            value == "dev" || value == "local"
        }
        Err(_) => false,
    }
}

/// Resolve a config value from the process environment, falling back to the
/// local `infra/.env` file (the same convenience the LLM client offers).
fn resolve_env(key: &str) -> Option<String> {
    if let Ok(value) = std::env::var(key) {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Some(value);
        }
    }
    env_from_files(key)
}

/// Look for `key` in the candidate `.env` file locations.
fn env_from_files(key: &str) -> Option<String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("GAIA_ENV_FILE") {
        candidates.push(PathBuf::from(path));
    }
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

/// Parse a minimal `.env` file into key/value pairs (mirrors `llm.rs`).
fn parse_dotenv(contents: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
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

// --- Wire types --------------------------------------------------------------

/// The query request body Cosmos expects.
#[derive(Debug, Serialize)]
struct QueryBody<'a> {
    query: &'a str,
    parameters: &'a [QueryParam],
}

/// The query response envelope; only the `Documents` array matters to us.
#[derive(Debug, serde::Deserialize)]
struct DocumentsResponse {
    #[serde(rename = "Documents")]
    documents: Vec<Record>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::RecordKind;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread::{self, JoinHandle};

    #[test]
    fn new_normalises_the_endpoint_to_end_with_a_slash() {
        let with = CosmosClient::new("https://acct.documents.azure.com:443/", "gaia", "tok");
        let without = CosmosClient::new("https://acct.documents.azure.com:443", "gaia", "tok");
        assert_eq!(with.endpoint(), without.endpoint());
        assert!(with.endpoint().ends_with('/'));
    }

    #[test]
    fn docs_url_targets_the_collection_documents_path() {
        let url = docs_url(
            "https://acct.documents.azure.com:443/",
            "gaia",
            "UsersDataLake",
        );
        assert_eq!(
            url,
            "https://acct.documents.azure.com:443/dbs/gaia/colls/UsersDataLake/docs"
        );
    }

    #[test]
    fn aad_auth_header_wraps_the_token_in_the_expected_envelope() {
        let header = aad_auth_header("abc.def.ghi");
        assert_eq!(header, "type%3Daad%26ver%3D1.0%26sig%3Dabc.def.ghi");
    }

    #[test]
    fn partition_key_header_is_a_json_array_and_escapes_quotes() {
        assert_eq!(partition_key_header("user-1"), "[\"user-1\"]");
        assert_eq!(partition_key_header("a\"b"), "[\"a\\\"b\"]");
    }

    #[test]
    fn rfc1123_formats_the_unix_epoch() {
        assert_eq!(rfc1123(0), "Thu, 01 Jan 1970 00:00:00 GMT");
    }

    #[test]
    fn rfc1123_formats_a_known_later_instant() {
        // 1_700_000_000 == 2023-11-14T22:13:20Z (a Tuesday).
        assert_eq!(rfc1123(1_700_000_000), "Tue, 14 Nov 2023 22:13:20 GMT");
    }

    #[test]
    fn parse_documents_reads_the_documents_envelope() {
        let body = r#"{
            "_rid": "abc",
            "Documents": [
                {"id": "d1", "userId": "user-1", "date": "2026-06-16", "data": "hello"},
                {"id": "d2", "entity": "rust", "date": "2026-06-15", "data": "world",
                 "dataVector": [0.1, 0.2]}
            ],
            "_count": 2
        }"#;

        let records = parse_documents(body).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].record_id, "d1");
        assert_eq!(records[0].user_id, "user-1");
        assert_eq!(records[1].entity_id, "rust");
        assert_eq!(records[1].data_vector, vec![0.1, 0.2]);
    }

    #[test]
    fn parse_documents_rejects_a_malformed_body() {
        assert!(matches!(
            parse_documents("not json"),
            Err(CosmosError::Decode(_))
        ));
    }

    #[test]
    fn query_param_accepts_strings_and_numbers() {
        let text = QueryParam::new("@text", "rust");
        let top = QueryParam::new("@top", 3_i64);
        assert_eq!(text.value, serde_json::json!("rust"));
        assert_eq!(top.value, serde_json::json!(3));
    }

    #[test]
    fn strip_quotes_only_removes_matching_pairs() {
        assert_eq!(strip_quotes("\"hello\""), "hello");
        assert_eq!(strip_quotes("'hello'"), "hello");
        assert_eq!(strip_quotes("\"mismatch'"), "\"mismatch'");
        assert_eq!(strip_quotes("plain"), "plain");
    }

    #[test]
    fn parse_dotenv_reads_keys_skips_comments_and_strips_quotes() {
        let contents = "\
            # a comment\n\
            COSMOS_ENDPOINT=https://acct.documents.azure.com:443/\n\
            \n\
            COSMOS_DATABASE=\"gaia\"\n\
            BROKEN LINE WITHOUT EQUALS\n";

        let map = parse_dotenv(contents);

        assert_eq!(
            map.get("COSMOS_ENDPOINT").map(String::as_str),
            Some("https://acct.documents.azure.com:443/")
        );
        assert_eq!(map.get("COSMOS_DATABASE").map(String::as_str), Some("gaia"));
        assert!(!map.contains_key("BROKEN LINE WITHOUT EQUALS"));
    }

    #[test]
    fn cosmos_error_messages_are_descriptive() {
        assert!(CosmosError::MissingToken
            .to_string()
            .contains("COSMOS_AAD_TOKEN"));
        assert!(CosmosError::Token("nope".into())
            .to_string()
            .contains("managed-identity"));
        assert!(CosmosError::Http("HTTP 403".into())
            .to_string()
            .contains("403"));
        assert!(CosmosError::Decode("bad".into())
            .to_string()
            .contains("decode"));
    }

    #[test]
    fn static_credential_bearer_returns_the_token_verbatim() {
        let client = CosmosClient::new("https://acct.documents.azure.com", "gaia", "tok-123");
        assert_eq!(client.bearer().expect("static token"), "tok-123");
    }

    #[test]
    fn managed_identity_bearer_uses_a_fresh_cached_token_without_network() {
        // Seed the cache with a token that refreshes far in the future, so the
        // fast path returns it and never touches the identity endpoint.
        let client =
            CosmosClient::with_managed_identity("https://acct.documents.azure.com", "gaia", "res");
        if let Credential::ManagedIdentity { cache, .. } = &client.cred {
            *cache.lock().expect("lock") = Some(CachedToken {
                value: "cached-mi-token".to_string(),
                refresh_after: unix_now() + 3_600,
            });
        } else {
            panic!("expected a managed-identity credential");
        }
        assert_eq!(client.bearer().expect("cached token"), "cached-mi-token");
    }

    #[test]
    fn debug_redacts_the_token_and_shows_only_the_auth_kind() {
        let client = CosmosClient::new("https://acct.documents.azure.com", "gaia", "super-secret");
        let rendered = format!("{client:?}");
        assert!(!rendered.contains("super-secret"));
        assert!(rendered.contains("static"));

        let mi =
            CosmosClient::with_managed_identity("https://acct.documents.azure.com", "gaia", "res");
        assert!(format!("{mi:?}").contains("managed-identity"));
    }

    #[test]
    fn cosmos_token_resource_strips_port_path_and_query() {
        assert_eq!(
            cosmos_token_resource("https://acct.documents.azure.com:443/"),
            "https://acct.documents.azure.com"
        );
        assert_eq!(
            cosmos_token_resource("https://acct.documents.azure.com/dbs/gaia"),
            "https://acct.documents.azure.com"
        );
        // A bare host with no scheme defaults to https.
        assert_eq!(
            cosmos_token_resource("acct.documents.azure.com"),
            "https://acct.documents.azure.com"
        );
    }

    #[test]
    fn rfc1123_formats_known_instants_with_std_only() {
        // The Unix epoch itself: 1970-01-01 was a Thursday at midnight.
        assert_eq!(rfc1123(0), "Thu, 01 Jan 1970 00:00:00 GMT");
        // A fixed later instant exercises the civil-from-days calendar maths and
        // the hour/minute/second split: 2026-06-16T12:34:56Z.
        //   days = 20_620 since epoch, plus 12:34:56 within the day.
        let secs = 20_620 * 86_400 + 12 * 3_600 + 34 * 60 + 56;
        assert_eq!(rfc1123(secs), "Tue, 16 Jun 2026 12:34:56 GMT");
    }

    #[test]
    fn civil_from_days_round_trips_known_dates() {
        // Day 0 is 1970-01-01; the algorithm must agree on the epoch and on a
        // leap-day boundary (2024-02-29 is day 19_782).
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    /// Spin up a one-shot local HTTP server that replies with `status` and
    /// `body`, returning its `http://127.0.0.1:<port>/` base URL and the join
    /// handle. Std-only (no extra dependency): it accepts a single connection,
    /// drains the request, and writes a fixed response.
    fn spawn_mock_cosmos(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Read the entire request before answering. Closing a socket that
                // still has unread bytes makes Windows send an RST, which the
                // client reports as "connection forcibly closed" while reading the
                // status line. Draining the request avoids that race.
                let mut data: Vec<u8> = Vec::new();
                let mut buf = [0u8; 1024];
                // First, read until we have the full header block (CRLF CRLF).
                let header_end = loop {
                    match stream.read(&mut buf) {
                        Ok(0) => break data.len(),
                        Ok(n) => {
                            data.extend_from_slice(&buf[..n]);
                            if let Some(pos) = find_subsequence(&data, b"\r\n\r\n") {
                                break pos + 4;
                            }
                        }
                        Err(_) => break data.len(),
                    }
                };
                // Then read any Content-Length body so nothing is left unread.
                let headers = String::from_utf8_lossy(&data[..header_end.min(data.len())]);
                let content_len = headers
                    .lines()
                    .find_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        lower
                            .strip_prefix("content-length:")
                            .map(|value| value.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                let mut body_seen = data.len().saturating_sub(header_end);
                while body_seen < content_len {
                    match stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => body_seen += n,
                        Err(_) => break,
                    }
                }
                // Write the fixed response and close the write half cleanly.
                let response = format!(
                    "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Write);
                // Drain until the client closes its side to dodge an RST on drop.
                loop {
                    match stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(_) => continue,
                        Err(_) => break,
                    }
                }
            }
        });
        (format!("http://{addr}/"), handle)
    }

    /// Find the first index of `needle` within `haystack`, or `None`.
    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    #[test]
    fn query_sends_the_request_and_parses_the_documents_envelope() {
        let body = r#"{"Documents":[{"id":"GaiaDiary|e|2026-05-10","entity":"e","date":"2026-05-10","data":"hello"}]}"#;
        let (endpoint, handle) = spawn_mock_cosmos("200 OK", body);
        let client = CosmosClient::new(endpoint, "gaia", "tok");

        let records = client
            .query(
                "GaiaDiary",
                "e",
                "SELECT * FROM c WHERE c.entity = @pk",
                &[QueryParam::new("@pk", "e")],
            )
            .expect("query succeeds against the mock");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].data, "hello");
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn query_maps_an_http_error_status_into_a_descriptive_error() {
        let (endpoint, handle) = spawn_mock_cosmos("403 Forbidden", "{\"message\":\"denied\"}");
        let client = CosmosClient::new(endpoint, "gaia", "tok");

        let err = client
            .query("GaiaDiary", "e", "SELECT 1", &[])
            .expect_err("a 403 is surfaced as an error");
        assert!(err.to_string().contains("403"), "got: {err}");
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn upsert_posts_the_record_and_succeeds_on_2xx() {
        let (endpoint, handle) = spawn_mock_cosmos("201 Created", "{}");
        let client = CosmosClient::new(endpoint, "gaia", "tok");
        let record = Record::new(
            "GaiaKB|e|2026-05-10",
            "e",
            "",
            "2026-05-10",
            RecordKind::KnowledgeBase,
            "payload",
            Vec::new(),
        );

        client
            .upsert("GaiaKB", "e", &record)
            .expect("upsert succeeds against the mock");
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn get_returns_the_record_on_200_and_none_on_404() {
        // 200: the document is returned.
        let doc = r#"{"id":"GaiaKB|e|2026-05-10","entity":"e","date":"2026-05-10","data":"hi"}"#;
        let (endpoint, handle) = spawn_mock_cosmos("200 OK", doc);
        let client = CosmosClient::new(endpoint, "gaia", "tok");
        let found = client
            .get("GaiaKB", "e", "GaiaKB|e|2026-05-10")
            .expect("get succeeds");
        assert_eq!(found.map(|r| r.data), Some("hi".to_string()));
        handle.join().expect("mock server thread joins");

        // 404: a miss is reported as Ok(None), not an error.
        let (endpoint, handle) = spawn_mock_cosmos("404 Not Found", "");
        let client = CosmosClient::new(endpoint, "gaia", "tok");
        let missing = client
            .get("GaiaKB", "e", "nope")
            .expect("404 is not an error");
        assert!(missing.is_none());
        handle.join().expect("mock server thread joins");
    }
}
