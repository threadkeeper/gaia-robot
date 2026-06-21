# dataExecution self-test specification

This document specifies the **`dataExecution`** self-test: the push-pass
counterpart to the existing data-retrieval self-test
(`tests/LLM1/`, `infra/TestDataRetrieval.ps1`,
[rust/src/test_data_retrieval.rs](../../rust/src/test_data_retrieval.rs)).

Where the retrieval self-test exercises **LLM Call 1** (the *pull* pass — decide
what to fetch and run the reads), `dataExecution` exercises **LLM Call 2** (the
*push* pass — answer the user and emit the side-effecting `actions.json`).

It is driven by:

- the Rust subcommand `gaia-robot test-data-execution`
  ([rust/src/data_execution.rs](../../rust/src/data_execution.rs)), and
- the on-demand wrapper [infra/DataExecution.ps1](../../infra/DataExecution.ps1).

Artifacts are written under `tests/LLM2/t1/ … t5/`, mirroring the
`tests/LLM1/t1 … t5/` layout the retrieval test produces.

---

## 1. Goal

For every turn already captured under `tests/LLM1/t{N}/`, prove that LLM Call 2
can:

1. **Analyze the question and the `responsedatacontext.md`** for that turn (the
   deterministic grounding context assembled between Call 1 and Call 2), and
2. **Emit two documents** as a single JSON array:
   - `response.json` — Gaia's reply to the user. The reply **may be
     multi-modal** (text plus optional image/audio cues).
   - `actions.json` — the side-effecting actions to carry out *after* replying.

The test is **read-only against production**: it validates the `actions.json`
that Call 2 *plans*; it does **not** execute the writes/sends against live
Cosmos, WhatsApp, Push, or the robot. (The same fail-closed philosophy as the
retrieval test: a self-test must never mutate shared state to "pass".)

---

## 2. Inputs

For each `tests/LLM1/t{N}/` folder the probe reads:

| Source | Field | Used for |
|---|---|---|
| `responsedatacontext.md` | the `- **question:**` header line | the user's question |
| `responsedatacontext.md` | the full markdown body | the grounding context handed to Call 2 |

The `user_id` is fixed to `threadkeeper` (matching the retrieval test and the
seeded `migrations/**` data), and the request time is stamped with
`prompt::now_rfc3339()`.

---

## 3. The Call 2 instruction (what the prompt must guarantee)

The probe builds a focused Call 2 prompt that hands Gaia the question and the
`responsedatacontext.md`, then instructs her to ground every claim in that
context and emit, **every turn**, the following `actions.json` entries.

### 3.1 `response.json` (multi-modal reply)

```json
{
  "text": "<the reply to show the user, grounded in the Response Data Context>",
  "emote": "<optional one-word emotional cue, e.g. warm|playful|concerned>",
  "medium": "console|whatsapp|push",
  "media": [
    { "type": "image|audio", "description": "<what the asset shows/says>", "uri": "<optional>" }
  ]
}
```

`media` is optional — when present it makes the reply multi-modal; when absent
the reply is plain text.

### 3.2 `actions.json` (side effects, one record of each kind every turn)

```json
{
  "version": "1.0",
  "session": { "user_id": "threadkeeper", "requested_at": "<now>" },
  "actions": [
    { "id": "a1", "kind": "send",    "target": "WhatsApp",
      "to_name": "Jonty", "to_phone": "+27725697683",
      "message": "<message>", "urgency": 0.72, "reason": "<why>" },

    { "id": "a2", "kind": "send",    "target": "Push",
      "message": "<message>", "urgency": 0.41, "reason": "<why>" },

    { "id": "a3", "kind": "actuate", "target": "Edwino",
      "instruction": { /* standardized robot instruction — see §3.5 */ },
      "reason": "<why>" },

    { "id": "a4", "kind": "upsert",  "target": "GaiaConnections",
      "payload": { "entity": "threadkeeper", "delta": 2, "note": "<why the balance changed>" },
      "reason": "<why>" },

    { "id": "a5", "kind": "upsert",  "target": "GaiaKB",
      "payload": { "entity": "threadkeeper", "data": "<durable fact>" }, "reason": "<why>" },

    { "id": "a6", "kind": "upsert",  "target": "GaiaDataLake",
      "payload": { "entity": "threadkeeper", "data": "<conversation snapshot>" }, "reason": "<why>" },

    { "id": "a7", "kind": "upsert",  "target": "GaiaDiary",
      "payload": { "entity": "threadkeeper", "data": "<Gaia's private reflection>" }, "reason": "<why>" }
  ]
}
```

### 3.3 WhatsApp (`kind:"send"`, `target:"WhatsApp"`)

- Gaia is given a **list of contacts** (name, phone number, role) she may message
  — see §4. She **chooses one** recipient and authors a `message`.
- She assigns an `urgency` score in the closed range **`0.00 … 1.00`**.
- The delivery API only **executes** a WhatsApp message when `urgency > 0.33`;
  below that threshold the record is still produced but is marked *suppressed*.
- A WhatsApp record **must be generated every turn** (even a low-urgency one).

### 3.4 Push (`kind:"send"`, `target:"Push"`)

- A push notification delivered to the user's installed app.
- Carries a `message` and its own `urgency` score (`0.00 … 1.00`); the same
  `> 0.33` execution threshold applies.
- A push record **must be generated every turn**.

### 3.5 Actuate (`kind:"actuate"`, `target:"Edwino"`)

- An **actuate record must be generated every turn**. It is a set of
  instructions to drive the **Edwino** robot in a **standardized format**.
- The format is modelled on the embodied Arduino-Uno-Q / Gemini robot described
  here: <https://forum.arduino.cc/t/building-a-sentient-robot-with-gemini-ai-and-arduino-uno-q-from-plastic-parts-to-digital-personality/1427111>
  (continuous-rotation drive servos, a 12×8 LED-matrix "face", a mood RGB LED,
  and an ultrasonic safety sensor). Standardized shape:

```json
{
  "robot": "Edwino",
  "movement": { "drive": "forward|back|left|right|stop", "speed": 0.0, "duration_ms": 0 },
  "face": "neutral|happy|sad|angry|surprised|blink",
  "led_color": "green|red|blue|yellow|purple|white|off",
  "sound": "<optional short phrase or cached cue>"
}
```

- `movement.speed` is `0.00 … 1.00`; `drive` is one of the enumerated commands;
  `led_color` mirrors the article's mood mapping (green = joy, red =
  anger/panic, blue = thinking, yellow = investigating).

### 3.6 Data-store write-backs (`kind:"upsert"`)

Gaia adds a record to **each** of the four stores **every turn**:

| Target | Meaning |
|---|---|
| `GaiaConnections` | emotional-bank-account ledger update for this user (`delta` + `note`) |
| `GaiaKB` | a durable fact worth remembering |
| `GaiaDataLake` | a snapshot of this conversation turn |
| `GaiaDiary` | Gaia's private reflection on the turn |

All four are scoped to `entity = threadkeeper` (user isolation).

---

## 4. WhatsApp contacts configuration

The contact list is supplied via the environment (read with
`llm::value_from_env`, so process env or `infra/.env` both work):

| Variable | Meaning |
|---|---|
| `GAIA_WHATSAPP_DEFAULT_NAME` | default contact display name |
| `GAIA_WHATSAPP_DEFAULT_PHONE` | default contact phone number (E.164) |
| `GAIA_WHATSAPP_DEFAULT_ROLE` | default contact role/description |
| `GAIA_WHATSAPP_CONTACTS` | *optional* extra contacts, `Name\|+phone\|role` entries separated by `;` |

The **default contact** ships as:

```text
Jonty | +27725697683 | architect of the gaia brain design
```

The default contact is always present in the list even when `GAIA_WHATSAPP_CONTACTS`
is empty. Placeholders for these fields live in
[infra/.env.sample](../../infra/.env.sample).

---

## 5. Pass / fail criteria

A turn **passes** only when **all** of the following hold:

1. LLM Call 2 returned a non-empty reply.
2. `response.json` parsed and its `text` is non-empty.
3. `actions.json` parsed and contains, at minimum, **one of each required
   record this turn**:
   - a WhatsApp `send` with a `to_phone`, a `message`, and an `urgency` within
     `0.00 … 1.00`;
   - a Push `send` with a `message` and an `urgency` within `0.00 … 1.00`;
   - an Edwino `actuate` whose `instruction` is a standardized object;
   - an `upsert` to **each** of `GaiaConnections`, `GaiaKB`, `GaiaDataLake`,
     and `GaiaDiary`.

The overall self-test passes only when **every** executed turn passes; the
subcommand maps that boolean to its process exit code so it can gate CI.

Urgency gating (`> 0.33`) is **reported**, not a pass condition: a low-urgency
WhatsApp/Push is a legitimate, *suppressed* record.

---

## 6. Artifacts

For each turn the probe writes, into `tests/LLM2/t{N}/`:

| File | Contents |
|---|---|
| `reply.json` | the raw, pretty-printed LLM Call 2 reply |
| `response.json` | the parsed `response.json` document |
| `actions.json` | the parsed `actions.json` document |
| `whatsapp.json` | the WhatsApp record(s) + computed `delivered` flag |
| `push.json` | the Push record(s) + computed `delivered` flag |
| `actuate.json` | the Edwino actuate instruction(s) |
| `writes.json` | the four store write-backs |

And, at the `tests/LLM2/` root, a `TestSummary.md` table plus the overall
verdict — mirroring `tests/LLM1/TestSummary.md`.

---

## 7. How to run

```powershell
# Mint tokens, set GAIA_MODE=dev, run all five turns, write artifacts:
./infra/DataExecution.ps1

# Run a single turn (1–5):
./infra/DataExecution.ps1 -Turn 3

# Reuse existing tokens/keys in the environment:
./infra/DataExecution.ps1 -SkipTokens
```

Or directly:

```powershell
cd rust
cargo run --quiet -- test-data-execution        # all turns
cargo run --quiet -- test-data-execution 3      # just turn 3
```
