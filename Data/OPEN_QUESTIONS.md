# Open Questions — blocking scaffolding

Please confirm/decide each item. Recommendations are marked **(R)**.

### Q1 — `recordName` vs the system `id`
The 65 existing rows use pipe-delimited ids like `GaiaDataLake|threadkeeper|2026-06-14`.
- **(R)** Keep existing `id` untouched on the current snapshot containers and add
  `recordName = {entity}_{yyyyMMdd}` as a *separate* field; use
  `id == recordName` only for the brand-new `DataLakeIndex`.
- Alternative: migrate all ids to `{entity}_{yyyyMMdd}` (cleaner, but rewrites
  existing documents).

### Q2 — Append format for `data`
- **(R)** Plain-text, newline-delimited, time-stamped log (best for embedding
  quality); store the exact structured turn in the DataLake analytical store.
- Alternative: structured JSON array of turn objects inside the document.

### Q3 — `DataLakeIndex` partition key & which DataLake feeds it
- **(R)** `/entity`, fed from `GaiaDataLake`. Add a `source` field so it can
  later also index `UsersDataLake`.
- Or: one index per DataLake, or a `/userId` partition for the user-scoped lake.

### Q4 — DataLake SQL endpoint (raw retrieval)
- Which technology: **Synapse Serverless SQL** over Cosmos analytical store, or
  **Microsoft Fabric** mirroring? (R) — whichever is already provisioned in
  `gaia-rg`; please confirm.
- Auth: SQL auth vs AAD/managed identity?
- This introduces a **new Rust SQL dependency** (e.g. `tiberius`). It must clear
  `cargo audit` / `cargo deny`. OK to add, with justification in the PR?

### Q5 — Embedding dimension override
- **(R)** Add a per-call `dimensions` override to `EmbeddingClient` and a new
  `DATALAKE_INDEX_VECTOR_DIMS=768` env var. Confirm 768 is the desired
  half-size (1536 / 2). `text-embedding-3-large` supports arbitrary `dimensions`.

### Q6 — Which containers get the append-and-re-embed writer?
Proposed: `GaiaKB`, `UsersKB`, `GaiaDataLake`, `UsersDataLake`, `GaiaDiary`
(in-place 1536-d vector for KB/Diary; DataLake uses the 768-d index instead).
`GaiaConnections` (ledger) and `GaiaWebSearchHistory` (audit log) are excluded.
Confirm?

### Q7 — Do DataLake daily records keep ANY Cosmos vector?
- **(R)** No — DataLake raw rows carry no `dataVector`; all DataLake semantic
  search goes through `DataLakeIndex` (768-d). Confirm this is the intent (vs.
  also keeping a 1536-d vector on the raw row).

### Q8 — Wiring point
The write path is currently not connected to live Cosmos (only `actions.json` is
emitted). Confirm we should wire `RecordWriter` into the engine push pass now, or
keep it behind the existing dev-mode/self-test harness first.
