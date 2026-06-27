//! The [`WriteDataController`]: Gaia's shared **write pass** (daily persistence).
//!
//! This module is the single source of truth for the *persistence* half of a
//! turn. Where the [`crate::pull_data_controller`] runs LLM Call 1's retrieval
//! and the [`crate::push_data_controller`] audits LLM Call 2's planned side
//! effects, this controller actually **writes** Gaia's per-day snapshot records
//! back to Cosmos with the lowest reasonable overhead:
//!
//! 1. **Read** today's record for `(container, entity)` by its deterministic
//!    daily id (a single ~1 RU point read).
//! 2. **Append** the new turn chunk to the day's plain-text log.
//! 3. **Re-embed** the whole day's text once (1536-d), so the stored
//!    `dataVector` is always current.
//! 4. **Write back** the document with a single upsert.
//! 5. For `GaiaDataLake`, also **sync** a small [`DataLakeIndex`] entry that
//!    stores only the id plus a half-size (768-d) vector *derived* from the same
//!    embedding (truncate + re-normalize — no second embedding call).
//!
//! The design (and every resolved decision behind it) is written up in the
//! repository's `Data/` folder. Crucially, both the live cloud app
//! ([`crate::engine::Engine`], once wired) and the **data-persistence self-test**
//! ([`crate::test_data_persistance`]) drive this same controller, so the two can
//! never drift apart: the test exercises the exact code that runs in production.

// Several public items below (the raw-store trait/stub, some helpers) are part
// of the scaffolded write/retrieve surface and are not all wired into the engine
// yet. Mirrors the `#![allow(dead_code)]` already used in `crate::storage`.
#![allow(dead_code)]

use std::fmt;
use std::fmt::Write as _;

use serde::Serialize;

use crate::cosmos::{CosmosClient, CosmosError, QueryParam};
use crate::embeddings::{EmbeddingClient, EmbeddingError};
use crate::storage::{Record, RecordKind};

/// The Cosmos container that holds the small DataLake semantic index.
pub const DATALAKE_INDEX_CONTAINER: &str = "DataLakeIndex";

/// The partition-key path the `DataLakeIndex` container is created with when it
/// does not yet exist. It mirrors `GaiaDataLake`'s `/entity` partition so an
/// index entry always shares its source row's partition.
pub const DATALAKE_INDEX_PARTITION_PATH: &str = "/entity";

/// The DataLake container whose writes also feed [`DATALAKE_INDEX_CONTAINER`].
pub const DATALAKE_CONTAINER: &str = "GaiaDataLake";

/// The append-only friendship ledger container.
///
/// Unlike the [`WRITER_CONTAINERS`], this store is **not** a daily
/// append-and-re-embed snapshot: each turn that changes the balance writes one
/// new immutable row keyed by an ISO-8601 `timestamp`, carrying the signed
/// `changeAmount` and the running `previousBalance`/`newBalance`. It is written
/// through [`WriteDataController::append_connection_delta`], never `upsert_daily`.
pub const CONNECTIONS_CONTAINER: &str = "GaiaConnections";

/// The default dimensionality of the derived `DataLakeIndex` vector (half of the
/// 1536-d source embedding). Overridable via `DATALAKE_INDEX_VECTOR_DIMS`.
pub const DEFAULT_INDEX_DIMS: usize = 768;

/// The snapshot containers the writer is allowed to append-and-re-embed into.
///
/// `GaiaConnections` (an append-only ledger) and `GaiaWebSearchHistory` (an
/// audit log) are deliberately excluded — see the `Data/` spec, Decision Q6.
pub const WRITER_CONTAINERS: [&str; 5] = [
    "GaiaKB",
    "UsersKB",
    "GaiaDataLake",
    "UsersDataLake",
    "GaiaDiary",
];

/// What a single [`WriteDataController::upsert_daily`] call ended up doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteAction {
    /// No record existed for this day; a new one was created.
    Created,
    /// A record already existed; the new chunk was appended to it.
    Appended,
    /// The append produced no change (idempotent replay); nothing was written.
    Unchanged,
}

impl WriteAction {
    /// A short lowercase label for reports and artifacts.
    pub fn label(self) -> &'static str {
        match self {
            WriteAction::Created => "created",
            WriteAction::Appended => "appended",
            WriteAction::Unchanged => "unchanged",
        }
    }
}

/// A structured summary of one daily write, returned so callers (the engine and
/// the self-test) can report exactly what happened without re-deriving it.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteOutcome {
    /// The container that was written, e.g. `GaiaDataLake`.
    pub container: String,
    /// The deterministic daily id, e.g. `GaiaDataLake|threadkeeper|2026-06-27`.
    pub id: String,
    /// Whether the record was created, appended to, or left unchanged.
    pub action: WriteAction,
    /// The size in bytes of the day's full text after the append.
    pub data_bytes: usize,
    /// The dimensionality of the in-place `dataVector` that was written
    /// (`0` when the write was skipped as unchanged).
    pub vector_dims: usize,
    /// `true` when a `DataLakeIndex` entry was also written this call.
    pub index_synced: bool,
    /// The dimensionality of the derived index vector (`0` when not synced).
    pub index_dims: usize,
}

/// Errors that can occur while persisting a record.
#[derive(Debug)]
pub enum WriteError {
    /// The target container is not one of [`WRITER_CONTAINERS`].
    NotWriterContainer(String),
    /// The controller has no Cosmos and/or embedding client configured.
    Offline,
    /// A Cosmos read or write failed.
    Cosmos(CosmosError),
    /// Computing the embedding failed.
    Embedding(EmbeddingError),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::NotWriterContainer(name) => {
                write!(f, "'{name}' is not a writer-enabled container")
            }
            WriteError::Offline => write!(
                f,
                "write pass is offline (Cosmos and/or embedding client not configured)"
            ),
            WriteError::Cosmos(err) => write!(f, "Cosmos write failed: {err}"),
            WriteError::Embedding(err) => write!(f, "embedding failed: {err}"),
        }
    }
}

impl std::error::Error for WriteError {}

impl From<CosmosError> for WriteError {
    fn from(err: CosmosError) -> Self {
        WriteError::Cosmos(err)
    }
}

impl From<EmbeddingError> for WriteError {
    fn from(err: EmbeddingError) -> Self {
        WriteError::Embedding(err)
    }
}

/// The on-the-wire shape of a `DataLakeIndex` document.
///
/// It stores only enough to find and re-fetch the raw row: the source id, the
/// partition (`entity`), the day, the originating container, and the half-size
/// vector. It deliberately carries **no** `data` body.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct DataLakeIndexDoc {
    /// Same id as the source DataLake row (1:1 mapping).
    id: String,
    /// Partition key — identical to `GaiaDataLake`'s `/entity`.
    entity: String,
    /// The record day (`YYYY-MM-DD`).
    date: String,
    /// Which DataLake the raw row lives in, e.g. `GaiaDataLake`.
    source: String,
    /// The truncated + re-normalized half-size embedding.
    #[serde(rename = "indexVector")]
    index_vector: Vec<f32>,
}

/// A structured summary of one friendship-ledger append, returned so callers
/// (the engine) can report the running balance without re-querying it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionOutcome {
    /// The deterministic ledger id, e.g. `GaiaConnections|<entity>|<timestamp>`.
    pub id: String,
    /// The signed delta applied this turn (`+` gain, `-` loss).
    pub change_amount: f64,
    /// The running balance before this change.
    pub previous_balance: f64,
    /// The running balance after this change.
    pub new_balance: f64,
}

/// The on-the-wire shape of a `GaiaConnections` ledger row.
///
/// One immutable entry per balance change, keyed within its `/entity` partition
/// by the ISO-8601 `timestamp`. It carries no `data`/`dataVector`/`metadata` —
/// only the signed delta and the running balance before and after it.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct ConnectionLedgerDoc {
    /// Unique id `GaiaConnections|<entity>|<timestamp>`.
    id: String,
    /// Partition key — the subject/user whose friendship balance changed.
    entity: String,
    /// The ISO-8601 instant of this change; the per-partition unique key.
    timestamp: String,
    /// The signed delta applied this turn (`+` gain, `-` loss).
    #[serde(rename = "changeAmount")]
    change_amount: f64,
    /// The running balance before this change.
    #[serde(rename = "previousBalance")]
    previous_balance: f64,
    /// The running balance after this change.
    #[serde(rename = "newBalance")]
    new_balance: f64,
    /// Why the balance changed this turn.
    notes: String,
}

/// Gaia's shared daily-write controller.
///
/// Holds the same optional clients the engine uses; when either is absent the
/// controller is "offline" and [`upsert_daily`](WriteDataController::upsert_daily)
/// returns [`WriteError::Offline`] rather than silently dropping data.
#[derive(Debug, Clone)]
pub struct WriteDataController {
    cosmos: Option<CosmosClient>,
    embedder: Option<EmbeddingClient>,
    index_dims: usize,
}

impl WriteDataController {
    /// Build a controller from explicit parts (used by the engine and tests).
    pub fn new(
        cosmos: Option<CosmosClient>,
        embedder: Option<EmbeddingClient>,
        index_dims: usize,
    ) -> Self {
        Self {
            cosmos,
            embedder,
            index_dims,
        }
    }

    /// Build a controller from the environment.
    ///
    /// Reuses [`CosmosClient::from_env`] and [`EmbeddingClient::from_env`] so the
    /// write pass shares the exact configuration as the rest of dev/local mode.
    /// The index vector size comes from `DATALAKE_INDEX_VECTOR_DIMS` (default
    /// [`DEFAULT_INDEX_DIMS`]).
    pub fn from_env() -> Result<Self, WriteError> {
        let cosmos = CosmosClient::from_env().map_err(WriteError::Cosmos)?;
        let embedder = EmbeddingClient::from_env().map_err(WriteError::Embedding)?;
        let index_dims = crate::llm::value_from_env("DATALAKE_INDEX_VECTOR_DIMS")
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|dims| *dims > 0)
            .unwrap_or(DEFAULT_INDEX_DIMS);
        Ok(Self::new(cosmos, embedder, index_dims))
    }

    /// `true` when both a Cosmos and an embedding client are configured.
    pub fn is_online(&self) -> bool {
        self.cosmos.is_some() && self.embedder.is_some()
    }

    /// Read back a written daily [`Record`] for verification (used by the
    /// self-test). Returns `Ok(None)` when the record does not exist.
    pub fn read_record(
        &self,
        container: &str,
        entity: &str,
        id: &str,
    ) -> Result<Option<Record>, WriteError> {
        let cosmos = self.cosmos.as_ref().ok_or(WriteError::Offline)?;
        Ok(cosmos.get(container, entity, id)?)
    }

    /// Read back the length of a `DataLakeIndex` entry's `indexVector`, or
    /// `Ok(None)` when no index document exists for `id` (used by the self-test
    /// to confirm the half-size vector was synced).
    pub fn read_index_vector_len(
        &self,
        entity: &str,
        id: &str,
    ) -> Result<Option<usize>, WriteError> {
        Ok(self.read_index_vector(entity, id)?.map(|v| v.len()))
    }

    /// Read back a `DataLakeIndex` entry's full `indexVector`, or `Ok(None)` when
    /// no index document exists for `id`.
    ///
    /// Returns the actual `f32` components (not just the length) so callers — the
    /// persistence self-test in particular — can verify the synced half-size
    /// vector is *sound*: correctly sized, finite, and unit-normalized by the
    /// Matryoshka truncation step.
    pub fn read_index_vector(
        &self,
        entity: &str,
        id: &str,
    ) -> Result<Option<Vec<f32>>, WriteError> {
        let cosmos = self.cosmos.as_ref().ok_or(WriteError::Offline)?;
        let value = cosmos.get_value(DATALAKE_INDEX_CONTAINER, entity, id)?;
        Ok(value.and_then(|doc| {
            doc.get("indexVector")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|n| n.as_f64().map(|f| f as f32))
                        .collect()
                })
        }))
    }

    /// Append `chunk` to today's record for `(container, entity)`, refresh its
    /// embedding, and write it back — creating the record on the first write of
    /// the day. For `GaiaDataLake`, also sync the `DataLakeIndex` entry.
    ///
    /// `now_rfc3339` is the current instant as `YYYY-MM-DDTHH:MM:SSZ` (pass
    /// [`crate::prompt::now_rfc3339`] in production; a fixed value in tests). The
    /// date portion keys the record; the time portion stamps the appended line.
    pub fn upsert_daily(
        &self,
        container: &str,
        entity: &str,
        now_rfc3339: &str,
        chunk: &str,
    ) -> Result<WriteOutcome, WriteError> {
        if !is_writer_container(container) {
            return Err(WriteError::NotWriterContainer(container.to_string()));
        }
        // Both clients are required to persist a fresh, embedded record.
        let (cosmos, embedder) = match (self.cosmos.as_ref(), self.embedder.as_ref()) {
            (Some(cosmos), Some(embedder)) => (cosmos, embedder),
            _ => return Err(WriteError::Offline),
        };

        let (date, time_marker) = split_rfc3339(now_rfc3339);
        let id = daily_id(container, entity, date);

        // 1. Read today's record (cheap point read), if it exists.
        let existing = cosmos.get(container, entity, &id)?;
        let previous_data = existing
            .as_ref()
            .map(|r| r.data.clone())
            .unwrap_or_default();
        let previous_hash = existing
            .as_ref()
            .and_then(|r| r.metadata.get(CONTENT_HASH_KEY).cloned());
        let created = existing.is_none();

        // 2. Append the new chunk to the day's plain-text log.
        let new_data = append_timestamped(&previous_data, chunk, time_marker);
        let new_hash = content_hash(&new_data);

        // M1: idempotent replay — identical content, nothing to do.
        if Some(&new_hash) == previous_hash.as_ref() {
            return Ok(WriteOutcome {
                container: container.to_string(),
                id,
                action: WriteAction::Unchanged,
                data_bytes: new_data.len(),
                vector_dims: 0,
                index_synced: false,
                index_dims: 0,
            });
        }

        // 3. Re-embed the whole day's text once (1536-d).
        let full_vector = embedder.embed(&new_data)?;

        // 4. Write the record back with the refreshed vector + content hash.
        let record = build_record(
            container,
            entity,
            date,
            &id,
            &new_data,
            full_vector.clone(),
            &new_hash,
        );
        cosmos.upsert(container, entity, &record)?;

        // 5. For the DataLake, sync the half-size index entry (derived locally).
        let mut index_synced = false;
        let mut index_dims = 0;
        if container == DATALAKE_CONTAINER {
            let index_vector = truncate_and_renormalize(&full_vector, self.index_dims);
            index_dims = index_vector.len();
            let index_doc = DataLakeIndexDoc {
                id: id.clone(),
                entity: entity.to_string(),
                date: date.to_string(),
                source: container.to_string(),
                index_vector,
            };
            // The DataLakeIndex container is derived (not provisioned alongside
            // the seven primary stores), so create it on first write if missing
            // rather than failing the turn's persistence with a 404.
            cosmos.upsert_doc_creating_container(
                DATALAKE_INDEX_CONTAINER,
                entity,
                &index_doc,
                DATALAKE_INDEX_PARTITION_PATH,
            )?;
            index_synced = true;
        }

        Ok(WriteOutcome {
            container: container.to_string(),
            id,
            action: if created {
                WriteAction::Created
            } else {
                WriteAction::Appended
            },
            data_bytes: new_data.len(),
            vector_dims: full_vector.len(),
            index_synced,
            index_dims,
        })
    }

    /// Append one signed-delta entry to the `GaiaConnections` friendship ledger.
    ///
    /// Unlike [`upsert_daily`](WriteDataController::upsert_daily), the ledger is
    /// **append-only**: this reads the entity's current running balance (the
    /// newest row's `newBalance`, or `0` when the ledger is empty), adds
    /// `change_amount`, and writes a brand-new immutable row keyed by
    /// `now_rfc3339`. It carries no embedding, so only a Cosmos client is needed
    /// (no embedder); a missing client yields [`WriteError::Offline`].
    ///
    /// `entity` is the `/entity` partition (the authenticated user id, matching
    /// how the ledger is read back). Returns the new id plus the before/after
    /// balance so the caller can surface the running total.
    pub fn append_connection_delta(
        &self,
        entity: &str,
        now_rfc3339: &str,
        change_amount: f64,
        notes: &str,
    ) -> Result<ConnectionOutcome, WriteError> {
        let cosmos = self.cosmos.as_ref().ok_or(WriteError::Offline)?;

        // 1. Read the current running balance (the newest ledger row's
        //    newBalance). An empty ledger starts the running total at zero.
        let previous_balance = latest_connection_balance(cosmos, entity)?;
        let new_balance = previous_balance + change_amount;

        // 2. Append a new immutable row keyed by this instant. The timestamp is
        //    the per-partition unique key, so each change is its own document.
        let id = connection_id(entity, now_rfc3339);
        let doc = ConnectionLedgerDoc {
            id: id.clone(),
            entity: entity.to_string(),
            timestamp: now_rfc3339.to_string(),
            change_amount,
            previous_balance,
            new_balance,
            notes: notes.to_string(),
        };
        cosmos.upsert_doc(CONNECTIONS_CONTAINER, entity, &doc)?;

        Ok(ConnectionOutcome {
            id,
            change_amount,
            previous_balance,
            new_balance,
        })
    }
}

// --- Pure helpers (no network) ----------------------------------------------

/// Build the deterministic ledger id `GaiaConnections|{entity}|{timestamp}`.
///
/// The ISO-8601 `timestamp` is the per-partition unique key, so each balance
/// change is its own immutable row (mirroring the ids on the live ledger).
fn connection_id(entity: &str, timestamp: &str) -> String {
    format!("{CONNECTIONS_CONTAINER}|{entity}|{timestamp}")
}

/// Read the entity's current friendship balance from the ledger.
///
/// Returns the newest row's `newBalance`, or `0.0` when the ledger has no rows
/// yet for this partition. Projects a single field and reads one row, so the
/// query stays a cheap, single-partition lookup.
fn latest_connection_balance(cosmos: &CosmosClient, entity: &str) -> Result<f64, CosmosError> {
    let rows = cosmos.query_values(
        CONNECTIONS_CONTAINER,
        entity,
        "SELECT TOP 1 c.newBalance FROM c WHERE c.entity = @entity ORDER BY c.timestamp DESC",
        &[QueryParam::new("@entity", entity)],
    )?;
    let balance = rows
        .first()
        .and_then(|row| row.get("newBalance"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    Ok(balance)
}

/// The `metadata` key under which the daily content hash is stored, enabling the
/// idempotent-replay (M1) skip.
const CONTENT_HASH_KEY: &str = "contentHash";

/// `true` when `container` is one of the append-and-re-embed [`WRITER_CONTAINERS`].
pub fn is_writer_container(container: &str) -> bool {
    WRITER_CONTAINERS.contains(&container)
}

/// `true` for the user-partitioned (`/userId`) snapshot containers.
///
/// The `Users*` containers partition on `/userId`; the `Gaia*` containers
/// partition on `/entity`. This decides which field on the [`Record`] carries
/// the business key so the document matches its partition header.
pub fn is_user_partitioned(container: &str) -> bool {
    container.starts_with("Users")
}

/// Build the deterministic daily id `{container}|{entity}|{date}`.
///
/// `date` must already be `YYYY-MM-DD`. This mirrors the id convention already
/// present on the live snapshot rows, so the writer reuses (not replaces) it.
pub fn daily_id(container: &str, entity: &str, date: &str) -> String {
    format!("{container}|{entity}|{date}")
}

/// Split an RFC-3339 instant `YYYY-MM-DDTHH:MM:SSZ` into its `(date, time)`
/// halves: the `YYYY-MM-DD` day and the `HH:MM:SSZ` time-of-day marker.
///
/// Falls back gracefully if the input is shorter than expected, so a malformed
/// timestamp can never panic the write path.
fn split_rfc3339(now_rfc3339: &str) -> (&str, &str) {
    match now_rfc3339.split_once('T') {
        Some((date, time)) => (date, time),
        None => (now_rfc3339, ""),
    }
}

/// Append `chunk` to a day's plain-text log as a new time-stamped line.
///
/// The first write of the day seeds the log with `[<time>] <chunk>`; later
/// writes append `\n\n[<time>] <chunk>` so the record stays a readable,
/// chronologically-ordered transcript that embeds cleanly.
pub fn append_timestamped(existing: &str, chunk: &str, time_marker: &str) -> String {
    let line = format!("[{time_marker}] {chunk}");
    if existing.is_empty() {
        line
    } else {
        format!("{existing}\n\n{line}")
    }
}

/// Derive a smaller unit-length vector from a Matryoshka embedding.
///
/// `text-embedding-3-large` packs the most important meaning into the earliest
/// dimensions, so a valid `dims`-length embedding is the first `dims` components
/// **re-normalized** to unit length (cosine similarity assumes unit vectors).
/// If the input is shorter than `dims` the whole input is used; a zero-length
/// slice is returned unchanged to avoid dividing by zero.
pub fn truncate_and_renormalize(vector: &[f32], dims: usize) -> Vec<f32> {
    let take = dims.min(vector.len());
    let slice = &vector[..take];
    let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return slice.to_vec();
    }
    slice.iter().map(|x| x / norm).collect()
}

/// Compute a stable hex SHA-1 of `data`, used only as a change detector (never
/// for security). Identical text yields an identical hash, enabling the M1
/// idempotent-replay skip.
fn content_hash(data: &str) -> String {
    let digest = crate::sha1::digest(data.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        // Writing to a String is infallible, so the result can be ignored.
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Build the [`Record`] to upsert for a daily write, placing the business key in
/// the correct partition field and stamping the content hash into `metadata`.
fn build_record(
    container: &str,
    entity: &str,
    date: &str,
    id: &str,
    data: &str,
    data_vector: Vec<f32>,
    content_hash: &str,
) -> Record {
    let user_partitioned = is_user_partitioned(container);
    let kind = if container.contains("DataLake") {
        RecordKind::DataLake
    } else {
        RecordKind::KnowledgeBase
    };
    let (entity_id, user_id) = if user_partitioned {
        ("", entity)
    } else {
        (entity, "")
    };
    let mut record = Record::new(id, entity_id, user_id, date, kind, data, data_vector);
    record
        .metadata
        .insert(CONTENT_HASH_KEY.to_string(), content_hash.to_string());
    record
}

// --- DataLake raw store (stubbed this phase, see Data/ spec §5.6) ------------

/// One raw DataLake row as returned by the raw store.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRow {
    /// The source id, e.g. `GaiaDataLake|threadkeeper|2026-06-27`.
    pub id: String,
    /// The record day (`YYYY-MM-DD`).
    pub date: String,
    /// The full raw text body.
    pub data: String,
}

/// Fetch raw DataLake rows by id.
///
/// In production this will hit the DataLake **SQL endpoint** (Synapse Serverless
/// / Microsoft Fabric). That endpoint is not provisioned yet (Decision Q4), so
/// the only implementation today is [`StubRawStore`], which reads the rows back
/// out of Cosmos. The trait keeps the rest of the retrieval flow stable while
/// the real backend is built behind it.
pub trait DataLakeRawStore {
    /// Return the raw rows for `ids`, skipping any that cannot be found.
    fn fetch(&self, ids: &[String]) -> Result<Vec<RawRow>, WriteError>;
}

/// A stand-in [`DataLakeRawStore`] that reads raw rows straight from Cosmos.
///
/// It derives each row's partition from its id (`{container}|{entity}|{date}`)
/// and point-reads it, so the two-tier retrieval flow is fully exercisable with
/// no SQL driver dependency.
#[derive(Debug, Clone)]
pub struct StubRawStore {
    cosmos: Option<CosmosClient>,
}

impl StubRawStore {
    /// Build a stub store over an optional Cosmos client.
    pub fn new(cosmos: Option<CosmosClient>) -> Self {
        Self { cosmos }
    }
}

impl DataLakeRawStore for StubRawStore {
    fn fetch(&self, ids: &[String]) -> Result<Vec<RawRow>, WriteError> {
        let cosmos = self.cosmos.as_ref().ok_or(WriteError::Offline)?;
        let mut rows = Vec::with_capacity(ids.len());
        for id in ids {
            // The id encodes its own partition: {container}|{entity}|{date}.
            let Some((container, entity, _date)) = split_daily_id(id) else {
                continue;
            };
            if let Some(record) = cosmos.get(container, entity, id)? {
                rows.push(RawRow {
                    id: record.record_id,
                    date: record.date,
                    data: record.data,
                });
            }
        }
        Ok(rows)
    }
}

/// Split a daily id back into `(container, entity, date)`.
///
/// Returns `None` when the id is not the expected three-part, `|`-delimited
/// shape, so a malformed id is skipped rather than panicking.
fn split_daily_id(id: &str) -> Option<(&str, &str, &str)> {
    let mut parts = id.splitn(3, '|');
    let container = parts.next()?;
    let entity = parts.next()?;
    let date = parts.next()?;
    if container.is_empty() || entity.is_empty() || date.is_empty() {
        return None;
    }
    Some((container, entity, date))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_id_joins_container_entity_and_date_with_pipes() {
        let id = daily_id("GaiaDataLake", "threadkeeper", "2026-06-27");
        assert_eq!(id, "GaiaDataLake|threadkeeper|2026-06-27");
    }

    #[test]
    fn split_rfc3339_separates_date_and_time() {
        let (date, time) = split_rfc3339("2026-06-27T09:34:29Z");
        assert_eq!(date, "2026-06-27");
        assert_eq!(time, "09:34:29Z");
    }

    #[test]
    fn split_rfc3339_handles_missing_time() {
        let (date, time) = split_rfc3339("2026-06-27");
        assert_eq!(date, "2026-06-27");
        assert_eq!(time, "");
    }

    #[test]
    fn append_timestamped_seeds_then_appends() {
        let first = append_timestamped("", "hello", "09:00:00Z");
        assert_eq!(first, "[09:00:00Z] hello");

        let second = append_timestamped(&first, "world", "10:30:00Z");
        assert_eq!(second, "[09:00:00Z] hello\n\n[10:30:00Z] world");
    }

    #[test]
    fn truncate_and_renormalize_yields_unit_length_half_vector() {
        // 4-d input whose tail holds weight, truncated to 2-d.
        let v = vec![0.6_f32, 0.0, 0.8, 0.0];
        let out = truncate_and_renormalize(&v, 2);
        assert_eq!(out.len(), 2);
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        // First component was the only weight, so it normalizes to 1.0.
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncate_and_renormalize_uses_all_when_shorter_than_dims() {
        let v = vec![3.0_f32, 4.0];
        let out = truncate_and_renormalize(&v, 768);
        assert_eq!(out.len(), 2);
        // 3-4-5 triangle -> normalized to (0.6, 0.8).
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn truncate_and_renormalize_returns_zero_vector_unchanged() {
        let v = vec![0.0_f32, 0.0, 0.0];
        let out = truncate_and_renormalize(&v, 2);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn content_hash_is_stable_and_sensitive() {
        let a = content_hash("same text");
        let b = content_hash("same text");
        let c = content_hash("different text");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 40); // 20-byte SHA-1 rendered as hex
    }

    #[test]
    fn writer_container_membership_excludes_ledger_and_audit_log() {
        assert!(is_writer_container("GaiaKB"));
        assert!(is_writer_container("GaiaDataLake"));
        assert!(is_writer_container("GaiaDiary"));
        assert!(!is_writer_container("GaiaConnections"));
        assert!(!is_writer_container("GaiaWebSearchHistory"));
    }

    #[test]
    fn user_partitioning_follows_the_users_prefix() {
        assert!(is_user_partitioned("UsersDataLake"));
        assert!(is_user_partitioned("UsersKB"));
        assert!(!is_user_partitioned("GaiaDataLake"));
        assert!(!is_user_partitioned("GaiaKB"));
    }

    #[test]
    fn build_record_places_key_in_the_right_partition_field() {
        let gaia = build_record(
            "GaiaKB",
            "threadkeeper",
            "2026-06-27",
            "GaiaKB|threadkeeper|2026-06-27",
            "body",
            vec![0.1, 0.2],
            "hash123",
        );
        assert_eq!(gaia.entity_id, "threadkeeper");
        assert_eq!(gaia.user_id, "");
        assert_eq!(
            gaia.metadata.get("contentHash").map(String::as_str),
            Some("hash123")
        );

        let users = build_record(
            "UsersDataLake",
            "threadkeeper",
            "2026-06-27",
            "UsersDataLake|threadkeeper|2026-06-27",
            "body",
            vec![0.1, 0.2],
            "hash123",
        );
        assert_eq!(users.user_id, "threadkeeper");
        assert_eq!(users.entity_id, "");
        assert_eq!(users.kind, RecordKind::DataLake);
    }

    #[test]
    fn data_lake_index_doc_serializes_to_the_expected_shape() {
        let doc = DataLakeIndexDoc {
            id: "GaiaDataLake|threadkeeper|2026-06-27".to_string(),
            entity: "threadkeeper".to_string(),
            date: "2026-06-27".to_string(),
            source: "GaiaDataLake".to_string(),
            index_vector: vec![0.5, 0.25],
        };
        let value = serde_json::to_value(&doc).unwrap();
        assert_eq!(value["id"], "GaiaDataLake|threadkeeper|2026-06-27");
        assert_eq!(value["entity"], "threadkeeper");
        assert_eq!(value["date"], "2026-06-27");
        assert_eq!(value["source"], "GaiaDataLake");
        assert_eq!(value["indexVector"], serde_json::json!([0.5, 0.25]));
        // No raw data body lives in the index.
        assert!(value.get("data").is_none());
        assert!(value.get("dataVector").is_none());
    }

    #[test]
    fn split_daily_id_round_trips_and_rejects_malformed() {
        let parts = split_daily_id("GaiaDataLake|threadkeeper|2026-06-27");
        assert_eq!(parts, Some(("GaiaDataLake", "threadkeeper", "2026-06-27")));
        assert_eq!(split_daily_id("no-pipes-here"), None);
        assert_eq!(split_daily_id("Gaia||2026-06-27"), None);
    }

    #[test]
    fn upsert_daily_rejects_non_writer_containers() {
        let controller = WriteDataController::new(None, None, DEFAULT_INDEX_DIMS);
        let err = controller
            .upsert_daily(
                "GaiaConnections",
                "threadkeeper",
                "2026-06-27T09:00:00Z",
                "x",
            )
            .unwrap_err();
        assert!(matches!(err, WriteError::NotWriterContainer(_)));
    }

    #[test]
    fn upsert_daily_is_offline_without_clients() {
        let controller = WriteDataController::new(None, None, DEFAULT_INDEX_DIMS);
        let err = controller
            .upsert_daily("GaiaKB", "threadkeeper", "2026-06-27T09:00:00Z", "x")
            .unwrap_err();
        assert!(matches!(err, WriteError::Offline));
        assert!(!controller.is_online());
    }

    #[test]
    fn stub_raw_store_is_offline_without_cosmos() {
        let store = StubRawStore::new(None);
        let err = store.fetch(&["GaiaDataLake|threadkeeper|2026-06-27".to_string()]);
        assert!(matches!(err, Err(WriteError::Offline)));
    }

    #[test]
    fn upsert_daily_creates_the_record_and_syncs_the_datalake_index() {
        // Drive the full GaiaDataLake write path against mock servers:
        //   1. point read -> 404 (no record yet => Created)
        //   2. embed the day's text -> an 8-d vector
        //   3. upsert the record -> 201
        //   4. upsert the half-size DataLakeIndex entry -> 201
        // The Cosmos client makes three sequential calls (read, upsert, index),
        // so it gets a three-response sequence server; the embedder makes one.
        let (cosmos_url, cosmos_handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("404 Not Found".to_string(), "{}".to_string()),
            ("201 Created".to_string(), "{}".to_string()),
            ("201 Created".to_string(), "{}".to_string()),
        ]);
        let (embed_url, embed_handle) = crate::test_http::spawn_mock_http(
            "200 OK",
            r#"{"data":[{"index":0,"embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}]}"#,
        );

        let cosmos = CosmosClient::new(cosmos_url, "GaiaDB", "token");
        let embedder = EmbeddingClient::for_test(embed_url);
        // A 4-d index (half of the 8-d mock embedding) keeps the assertion clear.
        let controller = WriteDataController::new(Some(cosmos), Some(embedder), 4);

        let outcome = controller
            .upsert_daily(
                DATALAKE_CONTAINER,
                "threadkeeper",
                "2026-06-27T09:00:00Z",
                "User: hi\nGaia: hello",
            )
            .expect("the write path succeeds against the mocks");

        assert_eq!(outcome.action, WriteAction::Created);
        assert_eq!(outcome.id, "GaiaDataLake|threadkeeper|2026-06-27");
        assert_eq!(outcome.vector_dims, 8);
        // The DataLake also syncs a derived half-size index entry.
        assert!(outcome.index_synced);
        assert_eq!(outcome.index_dims, 4);

        cosmos_handle.join().expect("cosmos mock thread joins");
        embed_handle.join().expect("embed mock thread joins");
    }

    #[test]
    fn upsert_daily_skips_an_identical_replay_without_writing() {
        // When re-running today's write would reproduce byte-identical content,
        // the stored content hash matches and the controller short-circuits as
        // Unchanged — making only the single point-read call (no embed, no
        // upsert). We seed an existing record whose stored hash equals the hash
        // of the text the append will produce.
        let chunk = "User: hi\nGaia: hello";
        // The append seeds an empty log, so replaying yields exactly this text.
        let final_data = append_timestamped("", chunk, "09:00:00Z");
        let hash = content_hash(&final_data);
        let existing = serde_json::json!({
            "id": "UsersDataLake|threadkeeper|2026-06-27",
            "userId": "threadkeeper",
            "date": "2026-06-27",
            "data": "",
            "metadata": { "contentHash": hash },
        })
        .to_string();
        // Only the point read happens on the replay path: a single response.
        let (cosmos_url, cosmos_handle) =
            crate::test_http::spawn_mock_http_sequence(vec![("200 OK".to_string(), existing)]);
        // The embedder must never be called here; point it at an unroutable port
        // so any accidental request would fail loudly rather than pass silently.
        let cosmos = CosmosClient::new(cosmos_url, "GaiaDB", "token");
        let embedder = EmbeddingClient::for_test("http://127.0.0.1:9/".to_string());
        let controller = WriteDataController::new(Some(cosmos), Some(embedder), 4);

        let outcome = controller
            .upsert_daily(
                "UsersDataLake",
                "threadkeeper",
                "2026-06-27T09:00:00Z",
                chunk,
            )
            .expect("the replay path succeeds without writing");

        assert_eq!(outcome.action, WriteAction::Unchanged);
        assert_eq!(outcome.vector_dims, 0);
        assert!(!outcome.index_synced);
        cosmos_handle.join().expect("cosmos mock thread joins");
    }

    #[test]
    fn upsert_daily_appends_to_an_existing_record() {
        // A record already exists for today with different content, so the write
        // appends (created == false) and re-embeds. UsersKB is not the DataLake,
        // so no index entry is synced. Cosmos serves the read then the upsert.
        let existing = serde_json::json!({
            "id": "UsersKB|threadkeeper|2026-06-27",
            "userId": "threadkeeper",
            "date": "2026-06-27",
            "data": "[08:00:00Z] earlier note",
            "metadata": { "contentHash": "stale-hash" },
        })
        .to_string();
        let (cosmos_url, cosmos_handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), existing),
            ("200 OK".to_string(), "{}".to_string()),
        ]);
        let (embed_url, embed_handle) = crate::test_http::spawn_mock_http(
            "200 OK",
            r#"{"data":[{"index":0,"embedding":[0.3,0.4]}]}"#,
        );

        let cosmos = CosmosClient::new(cosmos_url, "GaiaDB", "token");
        let embedder = EmbeddingClient::for_test(embed_url);
        let controller = WriteDataController::new(Some(cosmos), Some(embedder), 1);

        let outcome = controller
            .upsert_daily(
                "UsersKB",
                "threadkeeper",
                "2026-06-27T09:00:00Z",
                "a new line",
            )
            .expect("the append path succeeds");

        assert_eq!(outcome.action, WriteAction::Appended);
        assert!(!outcome.index_synced);
        assert_eq!(outcome.vector_dims, 2);
        cosmos_handle.join().expect("cosmos mock thread joins");
        embed_handle.join().expect("embed mock thread joins");
    }

    #[test]
    fn append_connection_delta_adds_to_the_running_balance() {
        // The ledger already holds a balance, so the new row carries the prior
        // newBalance as previousBalance and applies the delta on top. Cosmos
        // serves the balance query, then accepts the append. No embedder is
        // needed (the ledger has no vector), so we configure none.
        let balance_query = r#"{"Documents":[{"newBalance":5.0}]}"#.to_string();
        let (cosmos_url, cosmos_handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), balance_query),
            ("201 Created".to_string(), "{}".to_string()),
        ]);
        let controller = WriteDataController::new(
            Some(CosmosClient::new(cosmos_url, "GaiaDB", "token")),
            None,
            4,
        );

        let outcome = controller
            .append_connection_delta(
                "threadkeeper",
                "2026-06-27T09:00:00Z",
                3.0,
                "shared a laugh",
            )
            .expect("the ledger append succeeds against the mocks");

        assert_eq!(
            outcome.id,
            "GaiaConnections|threadkeeper|2026-06-27T09:00:00Z"
        );
        assert_eq!(outcome.previous_balance, 5.0);
        assert_eq!(outcome.change_amount, 3.0);
        assert_eq!(outcome.new_balance, 8.0);
        cosmos_handle.join().expect("cosmos mock thread joins");
    }

    #[test]
    fn append_connection_delta_starts_from_zero_on_an_empty_ledger() {
        // An empty ledger returns no rows, so the running balance starts at zero
        // and a negative delta is recorded against it.
        let empty_query = r#"{"Documents":[]}"#.to_string();
        let (cosmos_url, cosmos_handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), empty_query),
            ("201 Created".to_string(), "{}".to_string()),
        ]);
        let controller = WriteDataController::new(
            Some(CosmosClient::new(cosmos_url, "GaiaDB", "token")),
            None,
            4,
        );

        let outcome = controller
            .append_connection_delta("threadkeeper", "2026-06-27T09:00:00Z", -2.0, "a slight")
            .expect("the first ledger append succeeds");

        assert_eq!(outcome.previous_balance, 0.0);
        assert_eq!(outcome.new_balance, -2.0);
        cosmos_handle.join().expect("cosmos mock thread joins");
    }

    #[test]
    fn append_connection_delta_is_offline_without_a_cosmos_client() {
        let offline = WriteDataController::new(None, None, 4);
        assert!(matches!(
            offline.append_connection_delta("e", "2026-06-27T09:00:00Z", 1.0, "note"),
            Err(WriteError::Offline)
        ));
    }

    #[test]
    fn read_record_returns_the_record_on_200_and_errors_when_offline() {
        let record = serde_json::json!({
            "id": "GaiaKB|threadkeeper|2026-06-27",
            "entity": "threadkeeper",
            "date": "2026-06-27",
            "data": "remembered fact",
        })
        .to_string();
        let (cosmos_url, handle) =
            crate::test_http::spawn_mock_http_sequence(vec![("200 OK".to_string(), record)]);
        let controller =
            WriteDataController::new(Some(CosmosClient::new(cosmos_url, "GaiaDB", "t")), None, 4);
        let got = controller
            .read_record("GaiaKB", "threadkeeper", "GaiaKB|threadkeeper|2026-06-27")
            .expect("read succeeds")
            .expect("a record is present");
        assert_eq!(got.data, "remembered fact");
        handle.join().expect("cosmos mock thread joins");

        // Without a Cosmos client the read is Offline rather than a panic.
        let offline = WriteDataController::new(None, None, 4);
        assert!(matches!(
            offline.read_record("GaiaKB", "e", "id"),
            Err(WriteError::Offline)
        ));
    }

    #[test]
    fn read_index_vector_parses_the_index_vector_and_its_length() {
        let doc = r#"{"id":"x","indexVector":[0.1,0.2,0.3,0.4]}"#;
        // Two reads (full vector then length) each make one point read.
        let (cosmos_url, handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), doc.to_string()),
            ("200 OK".to_string(), doc.to_string()),
        ]);
        let controller =
            WriteDataController::new(Some(CosmosClient::new(cosmos_url, "GaiaDB", "t")), None, 4);

        let vector = controller
            .read_index_vector("threadkeeper", "x")
            .expect("read succeeds")
            .expect("an index doc is present");
        assert_eq!(vector, vec![0.1_f32, 0.2, 0.3, 0.4]);

        let len = controller
            .read_index_vector_len("threadkeeper", "x")
            .expect("read succeeds");
        assert_eq!(len, Some(4));
        handle.join().expect("cosmos mock thread joins");
    }
}
