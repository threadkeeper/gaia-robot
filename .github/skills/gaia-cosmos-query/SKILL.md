---
name: gaia-cosmos-query
description: Turn natural-language requests into safe Queries for GaiaCosmos, with default TOP 3 output and entity-based retrieval.
---

# GaiaCosmos query skill

Use this skill for the general Cosmos container `GaiaCosmos`.

## Rules
- Convert the user’s request into a Cosmos DB NoSQL query for `GaiaCosmos`.
- Default to `TOP 3` results.
- Use `entity` filters whenever the prompt names a topic, object, or domain.
- Add date and text filters only when they improve relevance.

## Query pattern
```sql
SELECT TOP 3 *
FROM c
WHERE c.entity = @entity
  AND <relevant filters>
ORDER BY c.date DESC
```

## Guidance
- Prefer narrow entity-focused filters over broad scans.
- When the request is ambiguous, ask one clarifying question before producing the query.
