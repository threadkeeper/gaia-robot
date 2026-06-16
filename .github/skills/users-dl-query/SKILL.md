---
name: users-dl-query
description: Turn natural-language requests into safe Queries for the UsersDL container, with user isolation and a default TOP 3 limit.
---

# UsersDL query skill

Use this skill when the request targets the user data-lake records stored in `UsersDL`.

## Rules
- Translate the request into a Cosmos DB NoSQL query for `UsersDL`.
- Always scope the query to the passed `UserID`. Never allow global or mixed-user retrieval.
- Default to `TOP 3` unless the user explicitly wants more results.
- Add date-range, semantic, or text filters only when the prompt asks for them.

## Query pattern
```sql
SELECT TOP 3 *
FROM c
WHERE c.userId = @userId
  AND <relevant filters>
ORDER BY c.date DESC
```

## Safety guardrails
- Never query `UsersDL` without `c.userId = @userId`.
- If the caller does not provide a `UserID`, ask for it before generating the query.
- Treat every `UsersDL` request as strictly isolated to the caller’s user partition.
