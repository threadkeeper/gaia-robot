---
name: users-kb-query
description: Turn natural-language requests into safe Queries for the UsersKB container, with user isolation and a default TOP 3 limit.
---

# UsersKB query skill

Use this skill when the request targets the user knowledge-base records stored in `UsersKB`.

## Rules
- Translate the user’s request into a Cosmos DB NoSQL query for `UsersKB`.
- Always require a `userId` predicate; never return records across users.
- Default the result set to `TOP 3` unless the user asks for more.
- Prefer `WHERE c.userId = @userId` and add date or text filters when the request implies them.
- If the request mentions “latest”, “recent”, or “newest”, sort by `c.date DESC`.

## Query pattern
```sql
SELECT TOP 3 *
FROM c
WHERE c.userId = @userId
  AND <relevant filters>
ORDER BY c.date DESC
```

## Safety guardrails
- Reject any query shape that omits the `userId` filter.
- Treat the incoming `UserID` as the only allowed partition scope for this container.
- Do not use cross-user or global search patterns for `UsersKB`.
