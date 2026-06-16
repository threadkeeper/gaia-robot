---
name: gaia-lh-query
description: Turn natural-language requests into safe Queries for GaiaLH, with default TOP 3 output and logical entity filtering.
---

# GaiaLH query skill

Use this skill for the logical-history container `GaiaLH`.

## Rules
- Create a Cosmos query for `GaiaLH` that reflects the user’s intent.
- Default to `TOP 3` unless the user asks for more.
- Filter by `entity` when the request names a subject, person, or system.
- Use date ordering when the request is about recent history or chronology.

## Query pattern
```sql
SELECT TOP 3 *
FROM c
WHERE c.entity = @entity
  AND <relevant filters>
ORDER BY c.date DESC
```

## Guidance
- Keep the query narrow and deterministic.
- For historical questions, prefer date-based ordering and concise result limits.
