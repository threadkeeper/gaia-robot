# Specification — Generic Cosmos Upsert & DataLakeIndex Semantic Search

**Repository:** `threadkeeper/gaia-robot`
**Scope:** Rust write-path for Cosmos DB containers + new `DataLakeIndex`
two-tier retrieval design.
**Status:** DRAFT FOR REVIEW — no code is to be written until this document and
`OPEN_QUESTIONS.md` are approved.

---

## 1. Background — what exists today

A survey of the current codebase (June 2026) establishes the baseline this spec
builds on:

### 1.1 Cosmos client (`rust/src/cosmos.rs`)
- `CosmosClient` is a thin, immutable, AAD-token-authenticated client bound to
  one account + database (`gaia`). API version `2018-12-31`.
- It exposes exactly three data-plane operations:
  - `query(container, partition_value, sql, params)` — single-partition read.
  - `upsert(container, partition_value, &Record)` — **blind** insert-or-replace
    (`x-ms-documentdb-is-upsert: true`). It does **not** read first, append, or
    touch embeddings.
  - `get(container, partition_value, id)` — point read, `Ok(None)` on 404.
- API version `2018-12-31` predates the Cosmos **Partial Document Update
  (Patch)** API (needs `2020-07-15`+), so patch-append is not available without
  a version bump. See §6.

### 1.2 Record model (`rust/src/storage.rs`)
- `Record` serde-maps onto the Cosmos document shape:
  `id` ↔ `record_id`, `entity` ↔ `entity_id`, `userId` ↔ `user_id`, plus
  `date`, `data`, `dataVector` (`Vec<f32>`), `metadata`.
- **There is no `recordName` field today.**
- The `Repository`/`KnowledgeBaseTable`/`DataLakeTable` types are an **in-memory
  model only**; their `upsert` does a `BTreeMap::insert` (full replace). They do
  not append or re-embed and are not connected to live Cosmos.

### 1.3 Embeddings (`rust/src/embeddings.rs`)
- `EmbeddingClient::embed(text) -> Vec<f32>` calls Azure OpenAI
  (`text-embedding`, `text-embedding-3-large`, 1536-d).
- The output dimension is **fixed per client** from
  `EMBEDDING_DIMENSIONS` / `COSMOS_VECTOR_DIMS` (currently 1536). There is **no**
  way to request a second, smaller dimension (e.g. 768) for the index.
  `text-embedding-3-large` *does* support the `dimensions` request parameter, so
  a half-size vector is achievable (see §5.3).

### 1.4 Live container inventory (queried 2026-06-27, db `gaia`)

| Container | PK | Records | `id` format observed | Has `data`/vector? |
|-----------|----|---------|----------------------|--------------------|
| `GaiaKB` | `/entity` | 3 | `GaiaKB\|threadkeeper\|2026-06-16` | yes |
| `GaiaDataLake` | `/entity` | 31 | `GaiaDataLake\|threadkeeper\|2026-06-14` | yes |
| `GaiaWebSearchHistory` | `/entity` | 0 | — | n/a (audit log) |
| `UsersDataLake` | `/userId` | 31 | `UsersDataLake\|threadkeeper\|2026-06-14` | yes |
| `UsersKB` | `/userId` | 0 | — | yes |
| `GaiaDiary` | `/entity` | 9 | `GaiaDiary\|threadkeeper\|2026-05-31` | yes |
| `GaiaConnections` | `/entity` | 4 | `GaiaConnections\|threadkeeper\|2026-04-27T19:09:59Z` | **no** (ledger) |

Observations that constrain the design:
- Existing `id`s already encode `{Container}|{businessKey}|{dateOrTimestamp}` —
  i.e. there is **one document per business key per day** on the snapshot
  containers. That is exactly the "append within the day" grain this spec wants.
- `GaiaConnections` is an append-only **ledger** (one row per event, ISO-8601
  timestamp, no `data`/`dataVector`) and is therefore **excluded** from the
  append-and-re-embed flow.

### 1.5 Write path is not wired
LLM Call 2 emits `actions.json` (see `rust/src/push_data_controller.rs`), but
**no production code calls `CosmosClient::upsert`** today (only a unit test
does). This spec therefore also defines where the new writer plugs into the
engine push pass.

---

## 2. Goals & non-goals

### Goals
1. A **generic, reusable** Rust write API that, for the snapshot containers,
   performs: *read-if-exists → append new data → recompute embedding (where
   applicable) → write back*, with the **lowest reasonable overhead**.
2. Introduce a `recordName` field = **`{entity}_{yyyyMMdd}`** on every written
   record, giving a stable, human-readable daily natural key.
3. Keep vector embeddings **up to date** whenever a record's text changes.
4. Add a new Cosmos container **`DataLakeIndex`** holding only
   `recordName` + a **half-size (768-d)** vector, used as a cheap semantic index.
5. Define the **two-tier retrieval**: semantic search over `DataLakeIndex` →
   `recordName`s → fetch full raw rows from the DataLake **SQL endpoint**
   (Synapse Serverless / Fabric).

### Non-goals (this phase)
- Implementing the code (this is spec-only; scaffolding follows approval).
- Changing `GaiaConnections` ledger semantics.
- Building the Synapse/Fabric pipeline itself (we specify the contract Gaia
  relies on; provisioning is tracked separately in `infra/`).
- Backfilling old records (covered as a follow-up migration, §8).

---

## 3. `recordName` design

```
recordName = "{entity}_{yyyyMMdd}"
```

- `entity` = the business/partition key (the user for `Users*`, the subject
  entity for `Gaia*`). For Gaia today these coincide (`threadkeeper`).
- `yyyyMMdd` = the UTC date of the record (matches the existing per-day grain).
- Examples: `threadkeeper_20260627`.

Properties:
- **One record per entity per UTC day** → natural append target.
- Deterministic: the writer can compute it from `(entity, today)` with no read.
- Distinct from the Cosmos system `id`. **Open question (Q1)**: do we (a) keep the
  current pipe-delimited `id` and add `recordName` as a separate field, or
  (b) make `id == recordName` going forward? Recommendation: **(a)** for the
  existing snapshot containers (avoid breaking the 65 existing rows), and
  **`id == recordName`** for the brand-new `DataLakeIndex` container.

`Record` gains a new serialized field:
```rust
#[serde(rename = "recordName", default, skip_serializing_if = "String::is_empty")]
pub record_name: String,
```

---

## 4. Generic append-and-re-embed upsert

### 4.1 The chosen algorithm (baseline)

For a write to a snapshot container (`GaiaKB`, `UsersKB`, `GaiaDataLake`,
`UsersDataLake`, `GaiaDiary`):

```
fn upsert_daily(container, entity, today_utc, new_chunk):
    1. record_name = f"{entity}_{yyyyMMdd(today_utc)}"
    2. existing = cosmos.get(container, entity, id_for(record_name))   # 1 point read (~1 RU)
    3. doc = existing.unwrap_or_else(|| new empty daily record)
    4. doc.data = append(doc.data, new_chunk)                          # structured append (§4.3)
    5. if container needs an in-place vector (KB, Diary):
           doc.dataVector = embedder.embed(doc.data)                   # 1 embedding call
       # DataLake containers do NOT store a 1536 vector here; their vector
       # lives in DataLakeIndex (§5).
    6. doc.recordName = record_name
    7. cosmos.upsert(container, entity, &doc)                          # 1 write
    8. if container is a DataLake:
           update_data_lake_index(entity, today_utc, doc.data)        # §5.4
```

Per write this is **1 point read + (0–1) embedding + 1 upsert** (+ the index
update for DataLake). The point read by `id` within a known partition is the
cheapest possible Cosmos read (~1 RU).

### 4.2 Why this method (overhead analysis)

| Option | Reads | Embeds | Writes | Notes |
|--------|-------|--------|--------|-------|
| **A. Read → append → re-embed whole → write back** *(chosen)* | 1 (point) | 1 | 1 | Embedding always reflects the full current text. Re-embeds the *whole daily record* each append, but a single day's text is bounded, so cost is small and predictable. |
| B. Cosmos **Patch** append (no read) | 0 | 1* | 1 | Cheapest text append, but recomputing the embedding still needs the *full* text → forces a read anyway (*), erasing the saving. Also needs API version bump (§6). |
| C. One sub-doc per turn + nightly rollup | 0 | many | many | More documents, more vectors, more RU, eventual consistency complexity. |

**Decision:** Option **A** is the baseline because correctness ("keep embedding
up to date") dominates and the per-day record is small. Two cost mitigations are
specified:
- **M1 — content-hash skip.** Store `metadata.contentHash` (e.g. SHA-1 of
  `data`). If an append produces no change (idempotent replay), skip the
  embedding call and the write.
- **M2 — embed-on-DataLake-index-only.** DataLake daily records do **not** carry
  a 1536-d `dataVector` in Cosmos at all (they are retrieved by SQL, not vector
  search), so the only embedding for a DataLake write is the **768-d** index
  vector — half the cost of a full embedding.

### 4.3 Append semantics
- `data` is treated as an append-only, newline-delimited log of turn chunks for
  the day. Each appended chunk is prefixed with a UTC time marker, e.g.
  `\n\n[HH:MM:SSZ] {chunk}` so the daily record stays human-readable and the
  embedding sees coherent text.
- **Open question (Q2):** plain-text append vs. a structured JSON array under a
  `metadata`/separate field. Recommendation: plain-text log for embedding
  quality, with the raw structured turn also written to the DataLake analytical
  store for exact retrieval.

### 4.4 Error handling & idempotency
- All fallible steps return `Result<_, E>` and use `?` (no `unwrap`/`expect`
  outside tests) per repo standards.
- The flow is **idempotent on replay** via M1 (same chunk → same hash → no-op).
- A failed embedding must **not** lose the appended text: write order is
  data-first is unsafe for vector freshness, so we embed **before** the single
  upsert (step 5 before 7); if embedding fails we surface the error and do not
  write a half-updated doc.

---

## 5. `DataLakeIndex` container + two-tier retrieval

### 5.1 Rationale
The DataLake raw text can grow large and is best queried analytically. We keep
Cosmos lean by storing only a small semantic index there and offloading raw
retrieval to the DataLake **SQL endpoint** (Synapse Serverless SQL over the
Cosmos analytical store, or Microsoft Fabric mirroring).

### 5.2 Container definition
- **Name:** `DataLakeIndex`
- **Partition key:** `/entity` (mirrors the source DataLake's business key).
  **Open question (Q3):** confirm `/entity` vs `/userId` to match whichever
  DataLake (`GaiaDataLake` vs `UsersDataLake`) feeds it, or hold both with a
  `source` discriminator.
- **Document shape:**
  ```jsonc
  {
    "id": "threadkeeper_20260627",        // == recordName
    "recordName": "threadkeeper_20260627",
    "entity": "threadkeeper",             // partition key
    "date": "2026-06-27",
    "source": "GaiaDataLake",             // which DataLake the raw row lives in
    "indexVector": [/* 768 float32 */],   // half-size cosine embedding
    "_ts": 0                               // Cosmos system timestamp
  }
  ```
- **No raw `data`** is stored here — only enough to find and fetch the raw row.
- **Vector index:** DiskANN cosine on `/indexVector`, `dimensions: 768`.

### 5.3 Half-size embedding (768-d)
- `text-embedding-3-large` supports the `dimensions` request parameter, so we
  request `768` for index vectors (vs `1536` for KB/Diary in-place vectors).
- Implementation note for scaffolding phase: add either
  `EmbeddingClient::embed_with_dimensions(text, dims)` or construct a second
  `EmbeddingClient` configured to 768. Recommendation: a per-call override so a
  single client serves both sizes. (`COSMOS_VECTOR_DIMS` stays 1536; a new
  `DATALAKE_INDEX_VECTOR_DIMS=768` env var configures the index size.)

### 5.4 Keeping the index in sync
On every DataLake daily upsert (§4.1 step 8):
```
update_data_lake_index(entity, today, full_day_text):
    record_name = f"{entity}_{yyyyMMdd(today)}"
    vec768 = embedder.embed_with_dimensions(full_day_text, 768)
    index_doc = { id: record_name, recordName, entity, date, source, indexVector: vec768 }
    cosmos.upsert("DataLakeIndex", entity, &index_doc)   # blind upsert is fine; whole doc is small
```
The index always reflects the current full-day text (the index vector is
recomputed from the same `full_day_text` used for the raw record).

### 5.5 Retrieval flow (read path)
```
1. query_vec768 = embedder.embed_with_dimensions(user_query, 768)
2. hits = cosmos.query("DataLakeIndex", entity,
        "SELECT TOP @k c.recordName, c.date, c.source
         FROM c
         WHERE c.entity = @pk AND IS_DEFINED(c.indexVector)
         ORDER BY VectorDistance(c.indexVector, @queryVector, false,
                  {distanceFunction:'Cosine',dataType:'Float32'})",
        params)                                   # cheap: small vectors, small container
3. record_names = hits.map(recordName)
4. raw_rows = datalake_sql.fetch(record_names)    # Synapse/Fabric SQL endpoint, §5.6
5. hand raw_rows to LLM Call 2 as Response Data Context
```
This mirrors the existing semantic-query conventions in
`rust/src/prompt.rs` (`VectorDistance`, `IS_DEFINED`, partition-pinned `@pk`).

### 5.6 DataLake SQL endpoint contract
- Source of truth for raw rows: the DataLake (`GaiaDataLake`/`UsersDataLake`)
  surfaced via **Synapse Serverless SQL** over Cosmos analytical store, or
  **Fabric** mirroring of the Cosmos container.
- Gaia fetches by key: `SELECT data, date FROM datalake WHERE recordName IN (...)`.
- **Open question (Q4):** which endpoint (Synapse Serverless vs Fabric), its
  connection/auth (SQL auth vs AAD), and the Rust SQL driver (e.g. `tiberius`).
  This pulls in a new dependency and must clear the repo's supply-chain bar
  (`cargo deny`/`cargo audit`); to be justified in the scaffolding PR.

---

## 6. API version / Patch consideration
- Sticking with blind read-modify-write (Option A) means **no API-version bump
  is required**; `2018-12-31` is sufficient.
- If we later adopt Cosmos Patch (Option B) for non-embedded appends, bump to
  `2020-07-15`+. Tracked as a future optimization, not in scope now.

---

## 7. Proposed module layout (for the scaffolding phase)
Following the repo's "one type per file" + readable-`main` conventions:

| New file | Type | Responsibility |
|----------|------|----------------|
| `rust/src/record_writer.rs` | `RecordWriter` | The generic append-and-re-embed upsert (§4); owns a `CosmosClient` + `EmbeddingClient`. |
| `rust/src/data_lake_index.rs` | `DataLakeIndex` | Build/sync index docs and run the §5.5 semantic query. |
| `rust/src/data_lake_sql.rs` | `DataLakeSqlClient` | Fetch raw rows by `recordName` from Synapse/Fabric (§5.6). |
| (edit) `rust/src/storage.rs` | `Record` | Add `record_name` field (§3). |
| (edit) `rust/src/embeddings.rs` | `EmbeddingClient` | Add per-call `dimensions` override (§5.3). |
| (edit) `rust/src/cosmos.rs` | `CosmosClient` | (Maybe) a typed cross-`Record`/index upsert helper. |
| (edit) `rust/src/main.rs` / engine push pass | — | Wire `RecordWriter` into the write path that currently only emits `actions.json`. |

---

## 8. Migration of existing rows (follow-up)
- 65 existing snapshot rows lack `recordName`. A one-shot backfill
  (`migrations/`) will set `recordName = {entity}_{yyyyMMdd(date)}` and, for the
  DataLake rows, populate `DataLakeIndex` with 768-d vectors.
- Backfill is **idempotent** (recompute from `date`) and safe to re-run.

---

## 9. Testing plan (per repo standards — every logic path covered)
- **Unit (in each new module's `#[cfg(test)] mod tests`):**
  - `recordName` formatting (entity + UTC date, padding, month/day edges).
  - Append semantics: empty→first chunk; existing→appended; idempotent replay
    (M1 hash skip).
  - "needs in-place vector" decision per container (KB/Diary yes; DataLake no;
    Connections excluded).
  - Index doc construction (768-d length, `id == recordName`, no raw `data`).
  - Semantic query string builder (partition-pinned, `IS_DEFINED`, `TOP`).
- **Integration (`tests/`):**
  - Round-trip against a mock/stub Cosmos (existing tests stub HTTP) — read,
    append, re-embed, write back, then re-read shows appended text + fresh
    vector.
  - DataLakeIndex sync stays consistent with the DataLake record.
- Coverage must keep `cargo llvm-cov --fail-under-lines 80` green.

---

## 10. Acceptance criteria
1. A documented, tested `RecordWriter::upsert_daily` performing read → append →
   re-embed → write back with `recordName = {entity}_{yyyyMMdd}`.
2. Embeddings stay current for KB/Diary (in-place 1536-d) and for DataLake (via
   768-d `DataLakeIndex`).
3. `DataLakeIndex` container defined, synced on every DataLake write, and
   queryable by vector similarity returning `recordName`s.
4. A defined (even if stubbed initially) path to fetch raw rows by `recordName`
   from the DataLake SQL endpoint.
5. `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, and coverage gate all
   pass.
6. No new unjustified dependencies; any SQL driver clears `cargo audit` /
   `cargo deny`.

---

*See `OPEN_QUESTIONS.md` for the decisions blocking scaffolding.*
