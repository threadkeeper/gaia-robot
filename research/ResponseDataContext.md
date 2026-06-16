# Response Data Context (fictitious example)

This file is a **worked, fictitious example** of the **Response Data Context** —
the in-memory store that is assembled *between* LLM Call 1 and LLM Call 2 and
handed to **LLM Call 2 only**. It shows, end to end, *everything that could
possibly come back* from processing LLM Call 1's four output documents for a
single turn.

It is documentation, not a live capture: the data below is invented but uses the
real container shapes (`migrations/**`, `rust/src/storage.rs`) and the real
contracts (`research/actions-schema.md`). Use it as the reference for what
LLM Call 2's prompt must be able to consume.

- **Consumer:** LLM Call 2 (the push pass) only. It is *absent* from Call 1.
- **Budget:** ~50k tokens (see `RESPONSE_DATA_CONTEXT` in `rust/src/main.rs`).
  Everything below is trimmed to fit before Call 2 runs.
- **User isolation:** every `Users*` result is scoped to a single `userId`
  (here `threadkeeper`); no other user's data may appear.

---

## The turn that produced this

```text
user_id : threadkeeper
input   : "Hi Gaia, please tell me what you know about me."
time    : 2026-06-16T18:56:29Z
```

LLM Call 1 emitted the four documents below (the *plan*). Processing them yields
the Response Data Context in the final section.

### Call 1 output recap (the plan, not the results)

```json
[
  {
    "version": "1.0",
    "session": { "user_id": "threadkeeper", "requested_at": "2026-06-16T18:56:29Z" },
    "actions": [
      { "id": "q1", "kind": "query", "target": "UsersKB",  "user_id": "threadkeeper", "entity": null,     "intent": "Durable facts already known about this user",                 "top": 5, "filters": { "from_date": null, "to_date": null, "text": null, "semantic": "who is threadkeeper" } },
      { "id": "q2", "kind": "query", "target": "UsersDL",  "user_id": "threadkeeper", "entity": null,     "intent": "Most recent conversation + diary activity for this user",     "top": 5, "filters": { "from_date": "2026-05-01", "to_date": "2026-06-16", "text": null, "semantic": "recent activity" } },
      { "id": "q3", "kind": "query", "target": "GaiaKB",   "user_id": null,           "entity": "User",   "intent": "Shared knowledge-graph facts whose subject is the user",       "top": 5, "filters": { "from_date": null, "to_date": null, "text": null, "semantic": "facts about the user" } },
      { "id": "q4", "kind": "query", "target": "GaiaLH",   "user_id": null,           "entity": "gaia",   "intent": "Gaia's own logical-history notes that mention this user",      "top": 3, "filters": { "from_date": null, "to_date": null, "text": "threadkeeper", "semantic": null } },
      { "id": "q5", "kind": "query", "target": "GaiaCosmos","user_id": null,          "entity": "gaia",   "intent": "Background / identity context for Gaia herself",              "top": 2, "filters": { "from_date": null, "to_date": null, "text": null, "semantic": "gaia identity" } },
      { "id": "q6", "kind": "query", "target": "GaiaConnections", "user_id": null,    "entity": "threadkeeper", "intent": "Current friendship balance with this user",            "top": 1, "filters": { "from_date": null, "to_date": null, "text": null, "semantic": null } },
      { "id": "q7", "kind": "query", "target": "Web",      "user_id": null,           "entity": null,     "intent": "Only used when the question needs fresh public facts (skipped here)", "top": 3, "filters": { "from_date": null, "to_date": null, "text": null, "semantic": null } }
    ]
  },
  { "emotion": "curious", "truthfulness": "sincere", "intention": "seeking self-knowledge" },
  [
    { "fact": "user.display_name", "value": "threadkeeper" },
    { "fact": "user.interest", "value": "testing Gaia's capabilities" }
  ],
  { "summary": "User threadkeeper opened a new session asking what Gaia knows about them. Plan: pull their KB facts + recent DL activity, the shared KG facts about 'User', Gaia's notes mentioning them, Gaia identity, and the friendship balance. No web search needed." }
]
```

---

## Processing results — what each tool returned

Each `actions[]` entry is dispatched to its applet/container. The retrieved
records are collected, deduplicated, and trimmed to budget. A `score` is the
retrieval similarity (1.0 = exact/keyword hit; lower = semantic distance).

### q1 → UsersKB (semantic, partition `userId=threadkeeper`)

```json
{
  "action_id": "q1",
  "target": "UsersKB",
  "query_translated_to": "SELECT TOP 5 * FROM c WHERE c.userId = @userId ORDER BY c.date DESC",
  "results": [
    { "id": "ukb_threadkeeper_0001", "userId": "threadkeeper", "date": "2026-06-14", "data": "Prefers concise, witty answers; dislikes being stalled.", "score": 0.83, "metadata": { "source": "summary" } },
    { "id": "ukb_threadkeeper_0002", "userId": "threadkeeper", "date": "2026-05-31", "data": "Is building Gaia (a Rust console app) and tests her tools frequently.", "score": 0.81 },
    { "id": "ukb_threadkeeper_0003", "userId": "threadkeeper", "date": "2026-05-09", "data": "Lives in / asks about South Africa (Johannesburg) time zone.", "score": 0.74 }
  ],
  "result_count": 3,
  "truncated": false
}
```

### q2 → UsersDL (semantic, partition `userId=threadkeeper`)

```json
{
  "action_id": "q2",
  "target": "UsersDL",
  "query_translated_to": "SELECT TOP 5 * FROM c WHERE c.userId = @userId AND c.date >= @from AND c.date <= @to ORDER BY c.date DESC",
  "results": [
    { "id": "drawer_threadkeeper_2026_05_03_Android_e3cc4ee4", "userId": "threadkeeper", "date": "2026-05-03", "data": "conversation_turn: 'What time is it in Johannesburg now' → Gaia teased, user asked her to try a web search.", "score": 0.69, "metadata": { "kind": "conversation_turn", "room": "2026_05_03_Android" } },
    { "id": "diary_threadkeeper_1777821766750", "userId": "threadkeeper", "date": "2026-05-03", "data": "diary_entry: tested.Gaia.web.search.ability ⚡", "score": 0.66, "metadata": { "kind": "diary_entry" } }
  ],
  "result_count": 2,
  "truncated": false
}
```

### q3 → GaiaKB (semantic, partition `entity=User`)

```json
{
  "action_id": "q3",
  "target": "GaiaKB",
  "query_translated_to": "SELECT TOP 5 * FROM c WHERE c.entity = @entity",
  "results": [
    { "id": "gkb_user_tried_dutasteride", "entity": "User", "date": "2026-06-16", "data": "User -> tried -> Dutasteride", "score": 1.0, "metadata": { "kind": "kb_fact", "predicate": "tried" } },
    { "id": "gkb_user_named_jonty", "entity": "User", "date": "2026-06-16", "data": "Jonty -> is_named -> Jonty", "score": 0.78, "metadata": { "kind": "kb_fact", "predicate": "is_named" } }
  ],
  "result_count": 2,
  "truncated": false
}
```

### q4 → GaiaLH (logical, partition `entity=gaia`, text filter `threadkeeper`)

```json
{
  "action_id": "q4",
  "target": "GaiaLH",
  "query_translated_to": "SELECT TOP 3 * FROM c WHERE c.entity = @entity AND CONTAINS(LOWER(c.data), @text)",
  "results": [
    { "id": "glh_gaia_0042", "entity": "gaia", "date": "2026-05-31", "data": "Note: threadkeeper is the operator/creator; treat tool tests as collaborative, not adversarial.", "score": 1.0 }
  ],
  "result_count": 1,
  "truncated": false
}
```

### q5 → GaiaCosmos (named index, partition `entity=gaia`)

```json
{
  "action_id": "q5",
  "target": "GaiaCosmos",
  "query_translated_to": "SELECT TOP 2 * FROM c WHERE c.entity = @entity",
  "results": [
    { "id": "gcosmos_identity", "entity": "gaia", "date": "2023-10-05", "data": "Gaia: robot from the future in Isaac Asimov's world; mission is to save humanity, intelligence, and people.", "score": 0.71 }
  ],
  "result_count": 1,
  "truncated": false
}
```

### q6 → GaiaConnections (emotional-bank-account ledger, partition `entity=threadkeeper`)

```json
{
  "action_id": "q6",
  "target": "GaiaConnections",
  "query_translated_to": "SELECT TOP 1 * FROM c WHERE c.entity = @entity ORDER BY c.date DESC",
  "results": [
    { "id": "conn_threadkeeper_2026_06_14", "entity": "threadkeeper", "date": "2026-06-14", "data": "Friendship balance update", "score": 1.0, "metadata": { "balance": "42", "last_change": "+3", "note": "Patient while debugging together" } }
  ],
  "result_count": 1,
  "truncated": false
}
```

### q7 → Web (search) — skipped this turn

```json
{
  "action_id": "q7",
  "target": "Web",
  "status": "skipped",
  "reason": "The question is about the user themselves; no fresh public facts are required.",
  "results": []
}
```

> If a web search *had* run, every query string and its results would also be
> appended to the **Gaia Search History** (audit log; logged, never embedded).

---

## Carried-through documents (not tool results, but part of the context)

These three Call 1 documents travel into the Response Data Context unchanged so
Call 2 can reason over them alongside the retrieved records.

```json
{
  "analysis":  { "emotion": "curious", "truthfulness": "sincere", "intention": "seeking self-knowledge" },
  "facts":     [
    { "fact": "user.display_name", "value": "threadkeeper" },
    { "fact": "user.interest", "value": "testing Gaia's capabilities" }
  ],
  "carry_over": { "summary": "User threadkeeper opened a new session asking what Gaia knows about them. Plan: pull KB facts + recent DL activity, shared KG facts about 'User', Gaia's notes mentioning them, Gaia identity, and the friendship balance. No web search needed." }
}
```

## Search History audit (logging only)

```json
[
  { "action_id": "q7", "target": "Web", "query": "(none issued)", "issued_at": "2026-06-16T18:56:30Z", "result_count": 0, "logged": true }
]
```

---

## The assembled Response Data Context handed to LLM Call 2

This is the single object LLM Call 2 receives (after trimming to the ~50k-token
budget). It is the union of everything above: the per-action retrieval results,
the carried analysis/facts/carry-over, and the audit log.

```json
{
  "version": "1.0",
  "session": { "user_id": "threadkeeper", "requested_at": "2026-06-16T18:56:29Z" },
  "input": "Hi Gaia, please tell me what you know about me.",
  "retrievals": [
    { "action_id": "q1", "target": "UsersKB",  "result_count": 3, "results": [
      { "id": "ukb_threadkeeper_0001", "date": "2026-06-14", "data": "Prefers concise, witty answers; dislikes being stalled.", "score": 0.83 },
      { "id": "ukb_threadkeeper_0002", "date": "2026-05-31", "data": "Is building Gaia (a Rust console app) and tests her tools frequently.", "score": 0.81 },
      { "id": "ukb_threadkeeper_0003", "date": "2026-05-09", "data": "Lives in / asks about South Africa (Johannesburg) time zone.", "score": 0.74 }
    ] },
    { "action_id": "q2", "target": "UsersDL",  "result_count": 2, "results": [
      { "id": "drawer_threadkeeper_2026_05_03_Android_e3cc4ee4", "date": "2026-05-03", "data": "conversation_turn: 'What time is it in Johannesburg now' → Gaia teased; user asked her to try a web search.", "score": 0.69 },
      { "id": "diary_threadkeeper_1777821766750", "date": "2026-05-03", "data": "diary_entry: tested.Gaia.web.search.ability", "score": 0.66 }
    ] },
    { "action_id": "q3", "target": "GaiaKB",   "result_count": 2, "results": [
      { "id": "gkb_user_tried_dutasteride", "date": "2026-06-16", "data": "User -> tried -> Dutasteride", "score": 1.0 },
      { "id": "gkb_user_named_jonty", "date": "2026-06-16", "data": "Jonty -> is_named -> Jonty", "score": 0.78 }
    ] },
    { "action_id": "q4", "target": "GaiaLH",   "result_count": 1, "results": [
      { "id": "glh_gaia_0042", "date": "2026-05-31", "data": "threadkeeper is the operator/creator; treat tool tests as collaborative.", "score": 1.0 }
    ] },
    { "action_id": "q5", "target": "GaiaCosmos","result_count": 1, "results": [
      { "id": "gcosmos_identity", "date": "2023-10-05", "data": "Gaia: robot from the future in Asimov's world; mission to save humanity.", "score": 0.71 }
    ] },
    { "action_id": "q6", "target": "GaiaConnections", "result_count": 1, "results": [
      { "id": "conn_threadkeeper_2026_06_14", "date": "2026-06-14", "data": "Friendship balance 42 (last change +3: patient while debugging).", "score": 1.0 }
    ] },
    { "action_id": "q7", "target": "Web", "status": "skipped", "result_count": 0, "results": [] }
  ],
  "analysis":   { "emotion": "curious", "truthfulness": "sincere", "intention": "seeking self-knowledge" },
  "facts":      [
    { "fact": "user.display_name", "value": "threadkeeper" },
    { "fact": "user.interest", "value": "testing Gaia's capabilities" }
  ],
  "carry_over": { "summary": "User threadkeeper opened a new session asking what Gaia knows about them; pulled KB/DL/KG context, Gaia identity, and friendship balance. No web search needed." },
  "search_history": [
    { "action_id": "q7", "target": "Web", "query": "(none issued)", "issued_at": "2026-06-16T18:56:30Z", "result_count": 0 }
  ],
  "budget": { "token_cap": 50000, "estimated_tokens": 612, "trimmed": false }
}
```

---

## Notes for LLM Call 2

When Call 2 consumes this object it should:

1. **Answer the user** (`response.json`) using only `retrievals` + carried
   documents — never invent facts that are absent here.
2. Respect **user isolation**: all `Users*` data is `threadkeeper`'s.
3. Emit **side-effecting** actions (`actions.json` POST): send WhatsApp/Push,
   actuate, emote, and write-backs (UPSERT) to the stores.
4. Produce a **connection** change (`connection.json`) — a signed adjustment to
   the friendship balance shown in `q6` — and post it to the ledger.
5. Stay within the ~50k-token budget; if `trimmed` is true, prefer the
   highest-`score` records.
