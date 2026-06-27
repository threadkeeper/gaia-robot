# Data — Cosmos Upsert & DataLakeIndex Specification

This folder scopes out (specification only — **no code yet**) the generic
record-write machinery for Gaia's Cosmos DB containers and the new two-tier
DataLake semantic-search design.

> **Status:** DRAFT FOR REVIEW. Nothing here is scaffolded. Once @threadkeeper
> signs off on `SPECIFICATION.md` (and resolves the items in
> `OPEN_QUESTIONS.md`), we begin implementation.

## Contents

| File | Purpose |
|------|---------|
| `SPECIFICATION.md` | The full functional + technical specification. |
| `OPEN_QUESTIONS.md` | Decisions that need @threadkeeper's confirmation before scaffolding. |

## One-paragraph summary

Today the codebase has only a low-level `CosmosClient::upsert` (blind
insert-or-replace) and an in-memory `Repository` model; there is no helper that
reads an existing daily record, appends new content, refreshes its embedding,
and writes it back. This spec defines a generic **append-and-re-embed upsert**
keyed by a new `recordName = {entity}_{yyyyMMdd}` field, plus a new Cosmos
container **`DataLakeIndex`** that stores only `recordName` + a half-size (768-d)
vector. Semantic search runs cheaply against `DataLakeIndex`; the matching
`recordName`s are then used to pull the full raw rows from the DataLake's
analytical SQL endpoint (Synapse Serverless / Microsoft Fabric).
