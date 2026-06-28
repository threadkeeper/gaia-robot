# Query skills

This folder contains the prompt/skill definitions that guide the LLM to translate natural-language requests into Cosmos DB NoSQL queries.

## Included skills
- `users-dl-query` — `UsersDL`, user-isolated, `TOP 3` by default.
- `gaia-kb-query` — `GaiaKB`, entity-aware, `TOP 3` by default.
- `gaia-lh-query` — `GaiaLH`, entity-aware, `TOP 3` by default.
- `gaia-cosmos-query` — `GaiaCosmos`, entity-aware, `TOP 3` by default.
- `gaia-connections-query` — `GaiaConnections`, the emotional-bank-account ledger, entity-aware, ordered by most recent `timestamp`, `TOP 3` by default.

## Mandatory behavior
- Every query must be scoped to the correct business key.
- Every `Users*` query must include `userId = @userId`.
- Every generated query defaults to `TOP 3` unless the user explicitly asks for more.
