# actions.json contract

This file defines the exact JSON format that the LLM should emit for the first action-planning pass.

## Top-level structure

```json
{
  "version": "1.0",
  "session": {
    "user_id": "user-123",
    "requested_at": "2026-06-16T12:00:00Z"
  },
  "actions": [
    {
      "id": "q1",
      "kind": "query",
      "target": "UsersKB",
      "user_id": "user-123",
      "entity": "notes",
      "intent": "Find the most recent notes for this user",
      "top": 3,
      "query": "SELECT TOP 3 c.id, c.userId, c.date, c.data FROM c WHERE c.userId = @pk AND CONTAINS(LOWER(c.data), 'meeting') ORDER BY c.date DESC",
      "filters": {
        "from_date": "2026-06-01",
        "to_date": "2026-06-16",
        "text": "meeting",
        "semantic": "recent notes"
      }
    }
  ]
}
```

## Field rules

- `version`: string, recommended `"1.0"`.
- `session.user_id`: required for every request; this is the user partition key used for isolation.
- `actions`: array of one or more query actions.
- Each action requires:
  - `id`: stable string identifier.
  - `kind`: must be `"query"`.
  - `target`: one of `Web`, `UsersKB`, `UsersDL`, `GaiaKB`, `GaiaLH`, `GaiaCosmos`, `GaiaConnections`.
  - `intent`: natural-language explanation of what the query is meant to retrieve.
  - `top`: integer; default to `3` when omitted or `0`.
- `query`: optional. The exact **single read-only `SELECT`** Cosmos SQL the model authored for this action. The executor runs it verbatim, binding the `@pk` partition value, and rejects anything that is not a single `SELECT`. When absent or blank, the executor builds a query from the structured fields instead. Not used for `Web` actions.
- For `UsersKB` and `UsersDL`, `user_id` is mandatory and must match the session user.
- `entity`, `filters.from_date`, `filters.to_date`, `filters.text`, and `filters.semantic` are optional but should be included when the prompt implies them.

## Example: user-scoped lookup

```json
{
  "version": "1.0",
  "session": {
    "user_id": "user-123",
    "requested_at": "2026-06-16T12:00:00Z"
  },
  "actions": [
    {
      "id": "q1",
      "kind": "query",
      "target": "UsersDL",
      "user_id": "user-123",
      "entity": "activity",
      "intent": "Retrieve the three most recent activity records for this user",
      "top": 3,
      "filters": {
        "from_date": "2026-06-01",
        "to_date": "2026-06-16"
      }
    }
  ]
}
```

## Example: shared knowledge lookup

```json
{
  "version": "1.0",
  "session": {
    "user_id": "user-123",
    "requested_at": "2026-06-16T12:00:00Z"
  },
  "actions": [
    {
      "id": "q1",
      "kind": "query",
      "target": "GaiaKB",
      "user_id": null,
      "entity": "robotics",
      "intent": "Find the three most relevant Gaia knowledge items about robotics",
      "top": 3,
      "filters": {
        "text": "robotics"
      }
    }
  ]
}
```

## Execution contract

The consumer should:
1. Load this JSON into an in-memory structure.
2. Validate that `kind == "query"` and `target` is supported.
3. Ensure `Users*` actions always contain `user_id`.
4. Default `top` to `3` when omitted or `0`.
5. Execute the query against the correct in-memory table or repository.
6. Return the result set into the response model without leaking other users’ data.
