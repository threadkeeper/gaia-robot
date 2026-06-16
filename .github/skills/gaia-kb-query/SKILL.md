---
name: gaia-kb-query
description: Turn natural-language requests into safe Queries for GaiaKB, with default TOP 3 output and entity-focused filtering.
---

# GaiaKB query skill

Use this skill for the shared knowledge-base container `GaiaKB`.

## Rules
- Convert the user’s natural-language request into a Cosmos DB NoSQL query for `GaiaKB`.
- Default to `TOP 3` results.
- Use `entity` filters when the request references a specific subject or knowledge item.
- Add date filtering when a time range is implied by the prompt.

## Query pattern
```sql
SELECT TOP 3 *
FROM c
WHERE c.entity = @entity
  AND <relevant filters>
ORDER BY c.date DESC
```

## Guidance
- Prefer the most relevant entity filter first, then add text or semantic filters.
- If no entity is provided, return a conservative query that asks for the best match only after the user clarifies the subject.
