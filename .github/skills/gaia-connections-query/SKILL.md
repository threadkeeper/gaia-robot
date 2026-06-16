---
name: gaia-connections-query
description: Turn natural-language requests into safe Queries for GaiaConnections, the emotional-bank-account ledger, with default TOP 3 output and entity-based retrieval ordered by most recent change.
---

# GaiaConnections query skill

Use this skill for the `GaiaConnections` ledger: Gaia's per-entity "emotional
bank account". Each record is one signed change to a running friendship balance.

## Record shape
Each `GaiaConnections` document has these fields:

| Field             | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `entity`          | Entity / user the balance is tracked for (partition key).     |
| `timestamp`       | ISO 8601 instant of the change (uniqueness within `entity`).  |
| `changeAmount`    | Signed delta applied this turn (`+` gain, `-` loss).          |
| `previousBalance` | Balance before this change.                                   |
| `newBalance`      | Balance after this change (`previousBalance + changeAmount`). |
| `notes`           | Why the change was made.                                      |

## Rules
- Convert the user's request into a Cosmos DB NoSQL query for `GaiaConnections`.
- Always filter by `entity` — a ledger is meaningless without its account holder.
- Default to `TOP 3` results unless the user explicitly asks for more.
- Order by `c.timestamp DESC` so the most recent ledger entries come first
  (this container keys on a timestamp, not a daily `date`).
- Add `changeAmount` sign, time-range, or `notes` text filters only when the
  prompt asks for them.

## Query pattern (recent ledger entries)
```sql
SELECT TOP 3 *
FROM c
WHERE c.entity = @entity
  AND <relevant filters>
ORDER BY c.timestamp DESC
```

## Query pattern (current balance)
The current balance is the `newBalance` of the most recent entry:
```sql
SELECT TOP 1 c.entity, c.newBalance, c.timestamp
FROM c
WHERE c.entity = @entity
ORDER BY c.timestamp DESC
```

## Guidance
- For "is our friendship growing/shrinking?" questions, look at the sign and
  trend of `changeAmount` across recent entries, or compare `newBalance` over
  time.
- For "why did the balance change?" questions, return `notes` alongside
  `changeAmount`, `previousBalance`, and `newBalance`.
- If the request names no entity, ask which account holder before querying.
