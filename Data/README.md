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
| `DECISIONS.md` | The resolved design decisions (Q1–Q8) confirmed by @threadkeeper. |

## One-paragraph summary

Today the codebase has only a low-level `CosmosClient::upsert` (blind
insert-or-replace) and an in-memory `Repository` model; there is no helper that
reads an existing daily record, appends new content, refreshes its embedding,
and writes it back. This spec defines a generic **append-and-re-embed upsert**
keyed by the existing daily `id` (`{container}|{entity}|{yyyy-MM-dd}`), plus a
new Cosmos container **`DataLakeIndex`** that stores only that `id` + a half-size
(768-d) vector. Semantic search runs cheaply against `DataLakeIndex`; the
matching `id`s are then used to pull the full raw rows from the DataLake's
analytical SQL endpoint (Synapse Serverless / Microsoft Fabric — stubbed for
now). All resolved decisions live in `DECISIONS.md`.

## How the 768-d index vector is produced (in one breath)

`text-embedding-3-large` is a **Matryoshka** embedding: training packs the most
important meaning into the *earliest* dimensions. So the 768-d index vector is
simply **the first 768 numbers of the 1536-d embedding, re-normalized to unit
length** (divide each kept component by the length of the truncated slice, so
cosine similarity stays valid). That means **one embedding call yields both
vectors** — the full 1536-d goes on the row, the derived 768-d goes in
`DataLakeIndex` — with no extra API call, cost, or latency, and only a small
retrieval-quality trade-off. It is one tiny pure function:
`truncate_and_renormalize(&[f32], 768) -> Vec<f32>`.
