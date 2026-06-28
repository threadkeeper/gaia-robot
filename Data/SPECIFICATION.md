# Specification — Generic Cosmos Upsert & DataLakeIndex Semantic Search

**Repository:** `threadkeeper/gaia-robot`
**Scope:** Rust write-path for Cosmos DB containers + new `DataLakeIndex`
two-tier retrieval design.
**Status:** APPROVED FOR SCAFFOLDING — all open questions resolved
(see `DECISIONS.md`). This document reflects @threadkeeper's choices.

---

## 0. Executive summary (read this first)

We add a generic **append-and-re-embed upsert** for Gaia's daily snapshot
records and a small **`DataLakeIndex`** container for cheap semantic search.

- **Daily key:** reuse the existing Cosmos `id`
  (`{container}|{entity}|{yyyy-MM-dd}`) — no new `recordName` field.
- **Write flow (lowest overhead):** point-read today's record → append a
  timestamped plain-text chunk → re-embed the whole day's text **once** (1536-d)
  → write back. ≈ 1 read + 1 embed + 1 upsert.
- **Two vectors, one call:** the row keeps the full **1536-d** `dataVector`; the
  `DataLakeIndex` keeps a **768-d** vector *derived locally* from that same
  embedding (see §5.3) — no second API call.
- **Two-tier DataLake retrieval:** vector search over the small `DataLakeIndex`
  returns `id`s → fetch the full raw rows from the DataLake **SQL endpoint**
  (Synapse/Fabric), which is **stubbed** this phase.
- **Containers covered:** `GaiaKB`, `GaiaDataLake`,
  `GaiaDiary`. Excluded: `GaiaConnections` (ledger), `GaiaWebSearchHistory`
  (audit log).
- **Wiring:** the writer is connected into the engine push pass now (the path
  that today only emits `actions.json`).

**Why the 768-d vector is "free":** `text-embedding-3-large` is a **Matryoshka**
embedding — the most important meaning is packed into the *earliest* dimensions.
So the 768-d index vector is just **the first 768 components of the 1536-d
vector, re-normalized to unit length** (divide each kept component by the length
of the truncated slice so cosine similarity stays valid). One embedding call
therefore yields both sizes, at the cost of one tiny pure function
`truncate_and_renormalize(&[f32], 768) -> Vec<f32>`.

---

## 1. Background — what exists today

### 1.1 Cosmos client (`rust/src/cosmos.rs`)
- `CosmosClient`: thin, immutable, AAD-token client bound to one account + db
  (`gaia`). API version `2018-12-31`.
- Three data-plane ops only:
  - `query(container, partition_value, sql, params)` — single-partition read.
  - `upsert(container, partition_value, &Record)` — **blind** insert-or-replace.
    No read, append, or embedding.
  - `get(container, partition_value, id)` — point read, `Ok(None)` on 404.

### 1.2 Record model (`rust/src/storage.rs`)
- `Record` serde-maps onto the Cosmos doc: `id`↔`record_id`, `entity`↔`entity_id`,
  `userId`↔`user_id`, plus `date`, `data`, `dataVector` (`Vec<f32>`), `metadata`.
- The `Repository`/`*Table` types are an **in-memory model only** (full-replace
  `BTreeMap::insert`); not wired to live Cosmos.

### 1.3 Embeddings (`rust/src/embeddings.rs`)
- `EmbeddingClient::embed(text) -> Vec<f32>` → Azure OpenAI
  `text-embedding-3-large`, 1536-d (fixed by `EMBEDDING_DIMENSIONS` /
  `COSMOS_VECTOR_DIMS`).
- Matryoshka model → a valid 768-d vector is the renormalized first-768 slice
  (§5.3); no second call needed.

### 1.4 Live container inventory (queried 2026-06-27, db `gaia`)

| Container | PK | Records | `id` format observed | Gets the writer? |
|-----------|----|---------|----------------------|------------------|
| `GaiaKB` | `/entity` | 3 | `GaiaKB\|threadkeeper\|2026-06-16` | ✅ in-place 1536-d |
| `GaiaDataLake` | `/entity` | 31 | `GaiaDataLake\|threadkeeper\|2026-06-14` | ✅ 1536-d + 768-d index |
| `GaiaWebSearchHistory` | `/entity` | 0 | — | ❌ audit log |
| `GaiaDiary` | `/entity` | 9 | `GaiaDiary\|threadkeeper\|2026-05-31` | ✅ in-place 1536-d |
| `GaiaConnections` | `/entity` | 4 | `...\|2026-04-27T19:09:59Z` | ❌ ledger |

Existing `id`s already encode `{Container}|{businessKey}|{date}` — **one doc per
business key per day** — exactly the grain we append into, so we reuse `id` as
the daily key (Decision Q1). `GaiaConnections` is an append-only ledger (no
`data`/`dataVector`) and is excluded.

### 1.5 Write path is not wired
LLM Call 2 emits `actions.json` (`rust/src/push_data_controller.rs`); **no
production code calls `CosmosClient::upsert`** today. Per Decision Q8 the writer
is wired into the engine push pass as part of this work.

---

## 2. Goals & non-goals

### Goals
1. A generic, reusable writer doing *read-if-exists → append → re-embed → write
   back*, lowest reasonable overhead.
2. Use the existing Cosmos `id` (`{container}|{entity}|{yyyy-MM-dd}`) as the daily
   key, computed deterministically from `(container, entity, today_utc)`.
3. Keep vector embeddings current whenever a record's text changes.
4. Add `DataLakeIndex` (partition `/entity`, like `GaiaDataLake`) holding only
   `id` + a 768-d vector.
5. Two-tier DataLake retrieval: vector search over `DataLakeIndex` → `id`s →
   raw rows from the DataLake SQL endpoint (**stubbed**, Decision Q4).

### Non-goals (this phase)
- Building the Synapse/Fabric pipeline or adding a SQL driver (stub only).
- Changing `GaiaConnections` semantics.
- Backfilling old rows (follow-up, §8).

---

## 3. Daily key (the `id`)

No new `recordName` field (Decision Q1). The existing Cosmos `id` is the daily
key, constructed in the established convention:

```
id = "{container}|{entity}|{yyyy-MM-dd}"      // e.g. GaiaDataLake|threadkeeper|2026-06-27
```

- `entity` = business/partition key (user for `Users*`, subject for `Gaia*`).
- `yyyy-MM-dd` = UTC date (matches existing rows).
- Deterministic → computed with no read, then point-read.
- `DataLakeIndex` docs use the **same `id`** as their source `GaiaDataLake` row.

No change to `Record`'s fields is required.

---

## 4. Generic append-and-re-embed upsert

### 4.1 Algorithm

```
fn upsert_daily(container, entity, today_utc, new_chunk):
    1. id  = "{container}|{entity}|{yyyy-MM-dd(today_utc)}"
    2. existing = cosmos.get(container, entity, id)          # 1 point read (~1 RU)
    3. doc = existing.unwrap_or_else(|| new empty daily record with this id/entity/date)
    4. doc.data = append_timestamped(doc.data, new_chunk)    # plain-text log (§4.3)
    5. if content hash unchanged:  return                    # M1 idempotent skip
    6. full = embedder.embed(doc.data)                       # ONE call, 1536-d
       doc.dataVector = full                                 # in-place vector for ALL writer containers
    7. cosmos.upsert(container, entity, &doc)                # 1 write
    8. if container == "GaiaDataLake":                       # DataLake also feeds the index
           v768 = truncate_and_renormalize(full, 768)        # derived, no extra API call
           index_doc = { id, entity, date, source: container, indexVector: v768 }
           cosmos.upsert("DataLakeIndex", entity, &index_doc)
```

Per write: **1 point read + 1 embedding + 1 upsert** (+ 1 small index upsert for
DataLake). Point read by `id` within a known partition is the cheapest read
(~1 RU).

> Decision Q7: DataLake rows keep the in-place **1536-d** `dataVector` **and** the
> `DataLakeIndex` holds the **768-d** vector. Decision Q5: the 768-d vector is
> *derived* from the same 1536-d embedding — still only **one** embedding call.

### 4.2 Why this method (overhead analysis)

| Option | Reads | Embeds | Writes | Notes |
|--------|-------|--------|--------|-------|
| **A. Read → append → re-embed whole → write back** *(chosen)* | 1 (point) | 1 | 1 (+1 tiny index) | Embedding always reflects full current text; a day's text is bounded so cost is small/predictable. |
| B. Cosmos **Patch** append (no read) | 0 | 1* | 1 | Cheapest text append, but re-embedding still needs the full text → forces a read anyway (*). Also needs an API-version bump. |
| C. One sub-doc per turn + nightly rollup | 0 | many | many | More docs/vectors/RU, eventual-consistency complexity. |

**Decision:** Option **A**. Mitigations:
- **M1 — content-hash skip.** Store `metadata.contentHash` (SHA-1 of `data`);
  idempotent replays skip embed + writes.
- **M2 — single embedding, two sizes.** Embed once at 1536-d; derive the 768-d
  index vector by truncation + re-normalization (§5.3).

### 4.3 Append semantics (Decision Q2)
- `data` is an append-only, newline-delimited, **timestamped plain-text log**.
  Each chunk appends as `\n\n[HH:MM:SSZ] {chunk}` — human-readable and good for
  embedding quality.
- First write of the day creates the record; later writes append.

### 4.4 Error handling & idempotency
- Fallible steps return `Result<_, E>` and use `?` (no `unwrap`/`expect` outside
  tests).
- Idempotent on replay via M1.
- Embed **before** the single upsert; on embedding failure, surface the error and
  never write a half-updated doc (text without a fresh vector).

---

## 5. `DataLakeIndex` container + two-tier retrieval

### 5.1 Rationale
DataLake raw text grows and is best queried analytically. Keep Cosmos lean: store
only a small semantic index, offload raw retrieval to the DataLake SQL endpoint
(stubbed this phase).

### 5.2 Container definition
- **Name:** `DataLakeIndex`
- **Partition key:** `/entity` — identical to `GaiaDataLake` (Decision Q3).
- **Fed from:** `GaiaDataLake` (`source` field leaves room for other DataLake sources).
- **Document shape:**
  ```jsonc
  {
    "id": "GaiaDataLake|threadkeeper|2026-06-27",  // SAME id as the source row
    "entity": "threadkeeper",                       // partition key
    "date": "2026-06-27",
    "source": "GaiaDataLake",
    "indexVector": [/* 768 float32 */]              // truncated+renormalized 1536-d embedding
  }
  ```
- No raw `data` here. **Vector index:** DiskANN cosine on `/indexVector`,
  `dimensions: 768`.

### 5.3 The 768-d vector: truncate + re-normalize (Decision Q5)
`text-embedding-3-large` is a Matryoshka model, so semantic mass concentrates in
the earliest dimensions. To get the index vector:

1. **Truncate** — take the first 768 components of the 1536-d embedding.
2. **Re-normalize** — the slice is no longer unit-length, and cosine similarity
   assumes unit vectors, so rescale:
   `norm = sqrt(Σ x_i²)` over the 768 kept components; `v[i] = slice[i] / norm`.

```
full_1536 = [ x0 … x767 | x768 … x1535 ]   (length 1.0)
slice_768 = full_1536[0..768]
norm      = sqrt(sum(slice_768[i]^2))
v768      = [ slice_768[i] / norm ]         (length 1.0 again)
```

One embedding call serves both the 1536-d row vector and the 768-d index vector.
Implemented as a pure, unit-tested helper `truncate_and_renormalize(&[f32], 768)`.
A new env var `DATALAKE_INDEX_VECTOR_DIMS=768` documents the size (default 768).

### 5.4 Keeping the index in sync
Updated inline in the DataLake write (§4.1 step 8) from the same embedding, so
the index always matches the raw row.

### 5.5 Retrieval flow (read path)
```
1. q1536 = embedder.embed(user_query)
   q768  = truncate_and_renormalize(q1536, 768)
2. hits = cosmos.query("DataLakeIndex", entity,
        "SELECT TOP @k c.id, c.date, c.source
         FROM c
         WHERE c.entity = @pk AND IS_DEFINED(c.indexVector)
         ORDER BY VectorDistance(c.indexVector, @queryVector, false,
                  {distanceFunction:'Cosine',dataType:'Float32'})",
        params)                                   # cheap: small vectors, small container
3. ids = hits.map(id)
4. raw_rows = datalake_sql.fetch(ids)             # STUBBED this phase (§5.6)
5. hand raw_rows to LLM Call 2 as Response Data Context
```
Mirrors existing conventions in `rust/src/prompt.rs` (`VectorDistance`,
`IS_DEFINED`, partition-pinned `@pk`).

### 5.6 DataLake SQL endpoint contract — STUBBED (Decision Q4)
- Endpoint not provisioned yet. Define a trait
  `DataLakeRawStore { fn fetch(&self, ids: &[String]) -> Result<Vec<RawRow>, _> }`
  and ship a **stub** (e.g. read back from `GaiaDataLake` by `id`, or return a
  marked placeholder) so the flow is exercisable.
- **No SQL driver dependency added this phase.** A real impl (Synapse/Fabric)
  later replaces the stub behind the same trait; any new dependency (e.g.
  `tiberius`) must clear `cargo audit` / `cargo deny`.

---

## 6. API version
- Option A needs **no API-version bump**; `2018-12-31` suffices. A future
  Patch-based optimization would require `2020-07-15`+ (out of scope).

---

## 7. Proposed module layout (scaffolding phase)
"One type per file" + readable-`main`:

| File | Type | Responsibility |
|------|------|----------------|
| `rust/src/record_writer.rs` | `RecordWriter` | Generic append-and-re-embed upsert (§4); owns `CosmosClient` + `EmbeddingClient`; computes the daily `id`. |
| `rust/src/data_lake_index.rs` | `DataLakeIndex` | Build/sync index docs (§5.4), run the §5.5 query, host `truncate_and_renormalize`. |
| `rust/src/data_lake_raw_store.rs` | `DataLakeRawStore` (trait) + `StubRawStore` | The §5.6 stubbed raw-fetch contract. |
| (edit) `rust/src/embeddings.rs` | `EmbeddingClient` | (Optional) place `truncate_and_renormalize` here; no size change needed. |
| (edit) `rust/src/engine.rs` / push pass | — | **Wire `RecordWriter` into the engine push pass** (Decision Q8). |

No change to `Record`'s fields is required.

---

## 8. Migration of existing rows (follow-up)
- The 31 `GaiaDataLake` rows already have a 1536-d `dataVector` but no
  `DataLakeIndex` entry. A one-shot, **idempotent** backfill (`migrations/`)
  derives the 768-d vector from each existing 1536-d vector (truncate +
  re-normalize — no re-embedding) and writes the index docs.
- Other writer containers keep their existing 1536-d vectors; no key change since
  we reuse the existing `id`.

---

## 9. Testing plan (per repo standards — every logic path covered)
- **Unit:**
  - Daily `id` construction (`{container}|{entity}|{yyyy-MM-dd}`, UTC, padding).
  - Append: empty→first; existing→appended; timestamp prefix; idempotent replay.
  - `truncate_and_renormalize`: length 768, unit norm, deterministic.
  - Index doc construction (`id == source id`, `/entity` PK, no raw `data`).
  - Semantic query builder (partition-pinned, `IS_DEFINED`, `TOP`).
  - `StubRawStore::fetch` returns rows for known ids, empties for unknown.
- **Integration (`tests/`):**
  - Round-trip on the HTTP-stub Cosmos: read → append → re-embed → write back;
    re-read shows appended text + fresh vector.
  - DataLake write produces a consistent 768-d `DataLakeIndex` doc.
  - Engine push pass performs the live upsert (wired path).
- Keep `cargo llvm-cov --fail-under-lines 80` green.

---

## 10. Acceptance criteria
1. `RecordWriter::upsert_daily` does read → append → re-embed → write back, keyed
   by the existing `id`, for the five writer containers.
2. KB/Diary and both DataLakes keep a current in-place 1536-d `dataVector`;
   `GaiaDataLake` writes also sync a 768-d `DataLakeIndex` doc derived from the
   same embedding.
3. `DataLakeIndex` (partition `/entity`) is queryable by vector similarity and
   returns source `id`s.
4. A **stubbed** `DataLakeRawStore` fetches raw rows by `id`, no SQL driver added.
5. `RecordWriter` is wired into the engine push pass and performs live writes.
6. `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, and the coverage gate
   all pass; no new unjustified dependencies.

---

*All decisions recorded in `DECISIONS.md`.*
