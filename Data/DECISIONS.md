# Resolved Decisions (2026-06-27, @threadkeeper)

These were the open questions; all are now decided. They are folded into
`SPECIFICATION.md`.

| # | Question | Decision |
|---|----------|----------|
| Q1 | `recordName` vs system `id` | **Drop the `recordName` concept.** Use the existing Cosmos `id` as the daily natural key everywhere. New records get an `id` in the existing convention `{container}\|{entity}\|{yyyy-MM-dd}`. |
| Q2 | Append format for `data` | **Plain-text, newline-delimited, timestamped log.** |
| Q3 | `DataLakeIndex` partition key / source | **Partition by `/entity`, identical to `GaiaDataLake`**, fed from `GaiaDataLake`. Keep an optional `source` field for future `UsersDataLake`. |
| Q4 | DataLake raw SQL endpoint | **Not provisioned yet — stub the interface.** No SQL driver dependency added in this phase. |
| Q5 | Index vector size & production | **768-d, derived by truncating + re-normalizing the 1536-d embedding** (Matryoshka). One embedding call per write. |
| Q6 | Which containers get the writer | **`GaiaKB`, `UsersKB`, `GaiaDataLake`, `UsersDataLake`, `GaiaDiary`.** Exclude `GaiaConnections` (ledger) and `GaiaWebSearchHistory` (audit log). |
| Q7 | Does the raw DataLake row keep a vector? | **Yes — keep the in-place 1536-d `dataVector` on the DataLake row, AND maintain the 768-d `DataLakeIndex`.** |
| Q8 | Wiring point | **Wire `RecordWriter` into the engine push pass now** (connect the currently-unwired write path to live Cosmos). |
