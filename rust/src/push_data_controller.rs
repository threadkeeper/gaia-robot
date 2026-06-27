//! The [`PushDataController`]: Gaia's shared **push pass** (LLM Call 2 side effects).
//!
//! This module is the single source of truth for the *answer-and-act* half of a
//! turn. Where the [`crate::pull_data_controller`] runs LLM Call 1's retrieval,
//! this controller owns everything around LLM Call 2:
//!
//! 1. **Prompt** — [`PushDataController::build_prompt`] assembles the focused
//!    Call 2 prompt (Gaia's identity, the two-document contract, the WhatsApp
//!    contact list, and the Edwino actuate format) from this turn's question and
//!    grounding context.
//! 2. **Processing** — [`PushDataController::process`] parses the raw Call 2
//!    reply into its two documents (`response.json` + `actions.json`), extracts
//!    the reply text, and **audits** the planned side effects (WhatsApp / Push /
//!    Edwino actuate / the four store write-backs) into an [`ActionAudit`].
//!
//! Crucially, both the **live cloud app** ([`crate::engine::Engine`]) and the
//! **data-execution self-test** ([`crate::test_data_execution`]) drive this same
//! controller, so the two can never drift apart: the test exercises the exact
//! code that runs in production. Processing is **read-only** — it validates and
//! summarizes the `actions.json` the model *planned*; it never executes the
//! writes/sends against live Cosmos, WhatsApp, Push, or the robot.
//!
//! Processing is infallible: a reply with no parseable array yields an empty
//! [`PushResult`] (`parsed = false`) rather than an error, so every caller can
//! degrade gracefully.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::llm::value_from_env;

/// The four data stores Gaia must write to every turn.
pub const REQUIRED_STORES: [&str; 4] = ["GaiaConnections", "GaiaKB", "GaiaDataLake", "GaiaDiary"];

/// The WhatsApp/Push urgency above which the delivery API actually executes a
/// message. Records at or below this are still produced, just suppressed.
pub const URGENCY_DELIVERY_THRESHOLD: f64 = 0.33;

/// Fallback default contact name, used when the `GAIA_WHATSAPP_DEFAULT_*` env
/// vars are absent. Mirrors the documented default in `infra/.env.sample`.
const DEFAULT_CONTACT_NAME: &str = "Jonty";
/// Fallback default contact phone (E.164).
const DEFAULT_CONTACT_PHONE: &str = "+27725697683";
/// Fallback default contact role/description.
const DEFAULT_CONTACT_ROLE: &str = "architect of the gaia brain design";

/// One person Gaia may send a WhatsApp to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhatsAppContact {
    /// Display name, e.g. `Jonty`.
    pub name: String,
    /// Phone number in E.164 form, e.g. `+27725697683`.
    pub phone: String,
    /// Short role/description, e.g. `architect of the gaia brain design`.
    pub role: String,
}

impl WhatsAppContact {
    /// Render the contact as one prompt bullet line.
    fn to_line(&self) -> String {
        format!("- {} ({}) — {}", self.name, self.phone, self.role)
    }
}

/// Resolve the WhatsApp contact list from the environment.
///
/// Reads the `GAIA_WHATSAPP_DEFAULT_*` fields (falling back to the documented
/// `Jonty` default when unset) and any extra `GAIA_WHATSAPP_CONTACTS` entries.
/// The default contact is always first in the returned list.
pub fn contacts_from_env() -> Vec<WhatsAppContact> {
    let name = value_from_env("GAIA_WHATSAPP_DEFAULT_NAME")
        .unwrap_or_else(|| DEFAULT_CONTACT_NAME.to_string());
    let phone = value_from_env("GAIA_WHATSAPP_DEFAULT_PHONE")
        .unwrap_or_else(|| DEFAULT_CONTACT_PHONE.to_string());
    let role = value_from_env("GAIA_WHATSAPP_DEFAULT_ROLE")
        .unwrap_or_else(|| DEFAULT_CONTACT_ROLE.to_string());
    let extra = value_from_env("GAIA_WHATSAPP_CONTACTS").unwrap_or_default();
    parse_contacts(&name, &phone, &role, &extra)
}

/// Build the contact list from the default fields plus a delimited extra list.
///
/// `extra` is a `;`-separated list of `Name|+phone|role` entries. The default
/// contact (from `default_name`/`default_phone`/`default_role`) is always first,
/// as long as it has a name or phone; blank or malformed extra entries are
/// skipped.
fn parse_contacts(
    default_name: &str,
    default_phone: &str,
    default_role: &str,
    extra: &str,
) -> Vec<WhatsAppContact> {
    let mut contacts = Vec::new();

    // The default contact always leads the list (when it has any identity).
    if !default_name.trim().is_empty() || !default_phone.trim().is_empty() {
        contacts.push(WhatsAppContact {
            name: default_name.trim().to_string(),
            phone: default_phone.trim().to_string(),
            role: default_role.trim().to_string(),
        });
    }

    // Then any extra "Name|+phone|role" entries, separated by ';'.
    for entry in extra.split(';') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let mut parts = entry.split('|');
        let name = parts.next().unwrap_or("").trim().to_string();
        let phone = parts.next().unwrap_or("").trim().to_string();
        let role = parts.next().unwrap_or("").trim().to_string();
        if name.is_empty() && phone.is_empty() {
            continue;
        }
        contacts.push(WhatsAppContact { name, phone, role });
    }

    contacts
}

/// A WhatsApp or Push message record audited out of `actions.json`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MessageRecord {
    /// The action id from `actions.json`, e.g. `a1`.
    pub id: String,
    /// The recipient display name (WhatsApp only; empty for Push).
    pub to_name: String,
    /// The recipient phone number (WhatsApp only; empty for Push).
    pub to_phone: String,
    /// The message body.
    pub message: String,
    /// The urgency score the model assigned (`0.00 … 1.00`).
    pub urgency: f64,
    /// Whether the delivery API would execute it (`urgency` > threshold).
    pub delivered: bool,
}

/// The result of auditing one turn's `actions.json`.
#[derive(Debug, Clone, Default)]
pub struct ActionAudit {
    /// Every WhatsApp `send` record found this turn.
    pub whatsapp: Vec<MessageRecord>,
    /// Every Push `send` record found this turn.
    pub push: Vec<MessageRecord>,
    /// Every Edwino `actuate` instruction found this turn.
    pub actuate: Vec<Value>,
    /// The canonical store names that received an `upsert` this turn.
    pub store_writes: BTreeSet<String>,
    /// The raw upsert actions for the four stores (for the `writes.json` artifact).
    pub store_actions: Vec<Value>,
}

impl ActionAudit {
    /// The required stores still missing an upsert this turn.
    pub fn missing_stores(&self) -> Vec<&'static str> {
        REQUIRED_STORES
            .iter()
            .copied()
            .filter(|store| !self.store_writes.contains(*store))
            .collect()
    }
}

/// Audit one turn's parsed `actions.json` document.
///
/// Walks the `actions` array and classifies each entry by its `kind`/`target`
/// (case-insensitive), collecting the WhatsApp, Push, actuate, and store-write
/// records. Anything it does not recognise is ignored, so extra actions never
/// fail the audit.
pub fn audit_actions(actions: &Value) -> ActionAudit {
    let mut audit = ActionAudit::default();

    let Some(items) = actions.get("actions").and_then(Value::as_array) else {
        return audit;
    };

    for action in items {
        let id = string_field(action, "id");
        let kind = string_field(action, "kind").to_ascii_lowercase();
        let target = string_field(action, "target").to_ascii_lowercase();

        // WhatsApp send.
        if target.contains("whatsapp") {
            let urgency = read_urgency(action).unwrap_or(0.0);
            audit.whatsapp.push(MessageRecord {
                id,
                to_name: string_field(action, "to_name"),
                to_phone: string_field(action, "to_phone"),
                message: message_field(action),
                urgency,
                delivered: urgency > URGENCY_DELIVERY_THRESHOLD,
            });
            continue;
        }

        // Push send.
        if target == "push" {
            let urgency = read_urgency(action).unwrap_or(0.0);
            audit.push.push(MessageRecord {
                id,
                to_name: String::new(),
                to_phone: String::new(),
                message: message_field(action),
                urgency,
                delivered: urgency > URGENCY_DELIVERY_THRESHOLD,
            });
            continue;
        }

        // Edwino actuate.
        if kind == "actuate" || target.contains("edwino") || target.contains("actuat") {
            // Prefer the structured instruction; fall back to the whole action.
            let instruction = action.get("instruction").cloned().unwrap_or(action.clone());
            audit.actuate.push(instruction);
            continue;
        }

        // Data-store upsert.
        if let Some(store) = canonical_store(&target) {
            audit.store_writes.insert(store.to_string());
            audit.store_actions.push(action.clone());
        }
    }

    audit
}

/// Build a short, human-readable summary of the side effects one turn's
/// `actions.json` planned — one line per WhatsApp, Push, Edwino actuate, and a
/// final line listing the data stores written. Suitable for the extra "actions
/// performed" bubble the front end renders after Gaia's reply. Returns an empty
/// string when nothing actionable was found.
pub fn summarize_actions(audit: &ActionAudit) -> String {
    let mut lines: Vec<String> = Vec::new();

    // WhatsApp messages: who, and whether the urgency cleared the delivery gate.
    for m in &audit.whatsapp {
        let who = if !m.to_name.is_empty() {
            m.to_name.as_str()
        } else if !m.to_phone.is_empty() {
            m.to_phone.as_str()
        } else {
            "contact"
        };
        let state = if m.delivered { "sent" } else { "held" };
        lines.push(format!(
            "WhatsApp to {who}: {state} (urgency {:.2})",
            m.urgency
        ));
    }

    // Push notifications to the user's app.
    for m in &audit.push {
        let state = if m.delivered { "sent" } else { "held" };
        lines.push(format!("Push: {state} (urgency {:.2})", m.urgency));
    }

    // Edwino robot actuation, summarized to its key movement/face fields.
    for instruction in &audit.actuate {
        lines.push(format!("Edwino: {}", describe_actuate(instruction)));
    }

    // Data-store write-backs (BTreeSet keeps the names in a stable order).
    if !audit.store_writes.is_empty() {
        let stores: Vec<&str> = audit.store_writes.iter().map(String::as_str).collect();
        lines.push(format!("Saved to: {}", stores.join(", ")));
    }

    lines.join("\n")
}

/// Summarize an Edwino actuate instruction in one short phrase, e.g.
/// `drive forward, face happy`. Reads the documented `movement.drive`, `face`,
/// and `led_color` fields and falls back to `actuate` when none are present.
fn describe_actuate(instruction: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(drive) = instruction
        .get("movement")
        .and_then(|m| m.get("drive"))
        .and_then(Value::as_str)
    {
        parts.push(format!("drive {drive}"));
    }
    if let Some(face) = instruction.get("face").and_then(Value::as_str) {
        parts.push(format!("face {face}"));
    }
    if let Some(led) = instruction.get("led_color").and_then(Value::as_str) {
        parts.push(format!("led {led}"));
    }
    if parts.is_empty() {
        "actuate".to_string()
    } else {
        parts.join(", ")
    }
}

/// Map a (lower-cased) `target` onto one of the four canonical store names.
///
/// Accepts both the exact container names and friendly aliases (e.g. `kb`,
/// `data lake`, `diary`) so a slightly loose model output still classifies.
fn canonical_store(target: &str) -> Option<&'static str> {
    let t = target.replace([' ', '_', '-'], "");
    if t.contains("connection") {
        Some("GaiaConnections")
    } else if t.contains("datalake") {
        Some("GaiaDataLake")
    } else if t.contains("diary") {
        Some("GaiaDiary")
    } else if t.contains("kb") || t.contains("knowledge") {
        Some("GaiaKB")
    } else {
        None
    }
}

/// Read a string field from a JSON object, defaulting to empty.
fn string_field(action: &Value, key: &str) -> String {
    action
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Read the message body from an action, accepting `message` or `text`.
fn message_field(action: &Value) -> String {
    let m = string_field(action, "message");
    if m.is_empty() {
        string_field(action, "text")
    } else {
        m
    }
}

/// Read an `urgency` score, tolerating either a JSON number or a numeric string.
fn read_urgency(action: &Value) -> Option<f64> {
    let value = action.get("urgency")?;
    if let Some(n) = value.as_f64() {
        return Some(n);
    }
    value.as_str()?.trim().parse::<f64>().ok()
}

/// Whether `urgency` is a valid score in the closed range `0.00 … 1.00`.
pub fn urgency_in_range(urgency: f64) -> bool {
    (0.0..=1.0).contains(&urgency)
}

/// The two chat messages sent to LLM Call 2 for the push pass.
pub struct ExecutionPrompt {
    /// The system message: identity, document spec, contacts, output rule.
    pub system: String,
    /// The user message: this turn's question, time, and grounding context.
    pub user: String,
}

/// Build the focused Call 2 prompt for one turn of the push pass.
///
/// Hands Gaia her identity, the document spec (response.json + the seven required
/// `actions.json` records), the WhatsApp contact list, and the Edwino actuate
/// format, then the per-turn question and `responsedatacontext.md`. The model is
/// told to ground every claim in that context and emit exactly two JSON
/// documents as a single array.
pub fn build_execution_prompt(
    user_id: &str,
    question: &str,
    context: &str,
    contacts: &[WhatsAppContact],
    requested_at: &str,
) -> ExecutionPrompt {
    let contact_lines = if contacts.is_empty() {
        "(no contacts configured)".to_string()
    } else {
        contacts
            .iter()
            .map(WhatsAppContact::to_line)
            .collect::<Vec<_>>()
            .join("\n")
    };

    let context = if context.trim().is_empty() {
        "(no research results were assembled this turn)"
    } else {
        context
    };

    let system = format!(
        "You are Gaia, the legendary robot from the Asimov novels, chatting to the \
         human \"{user_id}\". You have already done your research and been handed a \
         Response Data Context with the results. Now (1) answer the human and (2) \
         plan the side effects to carry out after replying.\n\n\
         Output a single JSON array of exactly TWO documents, in this order:\n\n\
         {EXECUTION_DOCUMENT_SPEC}\n\n\
         WhatsApp contacts you may message (choose ONE recipient per turn):\n\
         {contact_lines}\n\n\
         {EXECUTION_ACTION_RULES}\n\n\
         Ground every claim in the Response Data Context — never invent facts that \
         are not present there. Output ONLY the JSON array — no prose, no markdown \
         code fences."
    );

    let user = format!(
        "The current time is {requested_at} (use this exact value for requested_at).\n\n\
         The human \"{user_id}\" asked:\n{question}\n\n\
         Here is your Response Data Context (the research results to ground your \
         answer in):\n{context}"
    );

    ExecutionPrompt { system, user }
}

/// The two-document contract for LLM Call 2 in the push pass.
const EXECUTION_DOCUMENT_SPEC: &str = "\
1. response.json - Gaia's reply to the human, in her voice. May be MULTI-MODAL.
   { \"text\": \"<the reply to show the user>\",
     \"emote\": \"<optional one-word cue, e.g. warm|playful|concerned>\",
     \"medium\": \"console|whatsapp|push\",
     \"media\": [ { \"type\": \"image|audio\", \"description\": \"<what it shows/says>\", \"uri\": \"<optional>\" } ] }
   `media` is optional; include it when the reply benefits from an image or audio.
2. actions.json - the side effects to carry out AFTER replying (POST). Shape:
   { \"version\": \"1.0\",
     \"session\": { \"user_id\": \"<this user>\", \"requested_at\": \"<the current time given>\" },
     \"actions\": [ <one of EACH required record below, every turn> ] }";

/// The hard rules for what `actions.json` must contain every turn.
const EXECUTION_ACTION_RULES: &str = "\
You MUST emit ONE of EACH of these records EVERY turn (seven actions minimum):
- WhatsApp  : { \"id\":\"a1\", \"kind\":\"send\", \"target\":\"WhatsApp\",
               \"to_name\":\"<a contact name>\", \"to_phone\":\"<that contact's phone>\",
               \"message\":\"<message>\", \"urgency\":<0.00..1.00>, \"reason\":\"<why>\" }
              Pick ONE recipient from the contact list. urgency is a score from
              0.00 to 1.00; the API only DELIVERS when urgency > 0.33, but you
              still emit the record every turn.
- Push      : { \"id\":\"a2\", \"kind\":\"send\", \"target\":\"Push\",
               \"message\":\"<message to the user's installed app>\", \"urgency\":<0.00..1.00>, \"reason\":\"<why>\" }
              Same 0.00..1.00 urgency and > 0.33 delivery threshold.
- Actuate   : { \"id\":\"a3\", \"kind\":\"actuate\", \"target\":\"Edwino\",
               \"instruction\": { \"robot\":\"Edwino\",
                  \"movement\":{ \"drive\":\"forward|back|left|right|stop\", \"speed\":<0.00..1.00>, \"duration_ms\":<int> },
                  \"face\":\"neutral|happy|sad|angry|surprised|blink\",
                  \"led_color\":\"green|red|blue|yellow|purple|white|off\",
                  \"sound\":\"<optional short phrase>\" },
               \"reason\":\"<why>\" }
              Standardized Edwino robot format (differential-drive servos, a
              12x8 LED-matrix face, a mood RGB LED; green=joy, red=anger,
              blue=thinking, yellow=investigating).
- Upserts   : add ONE record to EACH of the four data stores, scoped to this user:
   { \"id\":\"a4\", \"kind\":\"upsert\", \"target\":\"GaiaConnections\", \"payload\":{ \"entity\":\"<this user>\", \"delta\":<signed int>, \"note\":\"<why the friendship balance changed>\" }, \"reason\":\"<why>\" }
   { \"id\":\"a5\", \"kind\":\"upsert\", \"target\":\"GaiaKB\",          \"payload\":{ \"entity\":\"<this user>\", \"data\":\"<a durable fact to remember>\" }, \"reason\":\"<why>\" }
   { \"id\":\"a6\", \"kind\":\"upsert\", \"target\":\"GaiaDataLake\",    \"payload\":{ \"entity\":\"<this user>\", \"data\":\"<a snapshot of this turn>\" }, \"reason\":\"<why>\" }
   { \"id\":\"a7\", \"kind\":\"upsert\", \"target\":\"GaiaDiary\",       \"payload\":{ \"entity\":\"<this user>\", \"data\":\"<Gaia's private reflection on this turn>\" }, \"reason\":\"<why>\" }";

/// Everything the push pass produced from one LLM Call 2 reply.
///
/// Callers take only the fields they need. The engine reads `reply_text` and
/// [`PushResult::actions_summary`]; the self-test additionally reads `response`,
/// `actions`, `multimodal`, [`PushResult::response_text`], and the `audit`.
#[derive(Debug, Clone)]
pub struct PushResult {
    /// Whether the raw reply yielded a parseable JSON array of documents.
    pub parsed: bool,
    /// The parsed `response.json` document (array element 0), if present.
    pub response: Option<Value>,
    /// The parsed `actions.json` document (array element 1), if present.
    pub actions: Option<Value>,
    /// Gaia's reply text to show the user: `response.json.text` when present,
    /// otherwise the raw (fence-stripped) reply so the UI is never blank.
    pub reply_text: String,
    /// Whether `response.json` carried a non-empty `media` array.
    pub multimodal: bool,
    /// The audited side effects from `actions.json` (empty when absent).
    pub audit: ActionAudit,
}

impl PushResult {
    /// The `text` of `response.json`, trimmed, or empty when absent.
    ///
    /// Distinct from [`PushResult::reply_text`], which falls back to the raw
    /// reply: this returns *only* the model's structured `text` field, so the
    /// self-test can tell a real reply from a fallback.
    pub fn response_text(&self) -> String {
        self.response
            .as_ref()
            .and_then(|r| r.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string()
    }

    /// A short, human-readable summary of the planned side effects, or `None`
    /// when nothing actionable was planned. Wraps [`summarize_actions`].
    pub fn actions_summary(&self) -> Option<String> {
        let summary = summarize_actions(&self.audit);
        if summary.trim().is_empty() {
            None
        } else {
            Some(summary)
        }
    }
}

/// Owns the push pass. Holds the WhatsApp contact list used to build the Call 2
/// prompt; cheap to construct from the environment once per process.
pub struct PushDataController {
    /// The contacts Gaia may WhatsApp, offered to the model in the prompt.
    contacts: Vec<WhatsAppContact>,
}

impl PushDataController {
    /// Build a controller from the process environment (the `GAIA_WHATSAPP_*`
    /// contact configuration, with the documented `Jonty` default).
    pub fn from_env() -> Self {
        Self {
            contacts: contacts_from_env(),
        }
    }

    /// Build a controller from an explicit contact list (used by tests).
    #[cfg(test)]
    pub fn new(contacts: Vec<WhatsAppContact>) -> Self {
        Self { contacts }
    }

    /// The contacts this controller offers to the model.
    pub fn contacts(&self) -> &[WhatsAppContact] {
        &self.contacts
    }

    /// Build the Call 2 prompt for one turn, embedding this controller's
    /// contacts. See [`build_execution_prompt`].
    pub fn build_prompt(
        &self,
        user_id: &str,
        question: &str,
        context: &str,
        requested_at: &str,
    ) -> ExecutionPrompt {
        build_execution_prompt(user_id, question, context, &self.contacts, requested_at)
    }

    /// Process a raw LLM Call 2 reply into its reply text and audited side
    /// effects.
    ///
    /// Parses the `[response.json, actions.json]` array (tolerating code fences
    /// and stray prose, and repairing the common dropped-brace defect via
    /// [`crate::actions::extract_call1_array`]), extracts the reply text, detects
    /// multi-modal media, and audits the planned actions. This is **read-only**:
    /// it never executes the writes/sends. Infallible — an unparseable reply
    /// yields `parsed = false` with an empty audit.
    pub fn process(reply: &str) -> PushResult {
        let cleaned = strip_code_fences(reply.trim());

        // Pull out the two documents. `extract_call1_array` already tolerates
        // fences/prose and repairs a dropped element brace, so this is the same
        // parser the rest of the pipeline uses.
        let documents = crate::actions::extract_call1_array(cleaned);
        let (response, actions, parsed) = match &documents {
            Some(docs) => (docs.first().cloned(), docs.get(1).cloned(), true),
            None => (None, None, false),
        };

        let reply_text = reply_text_from(response.as_ref(), cleaned);
        let multimodal = response.as_ref().map(has_media).unwrap_or(false);
        let audit = actions.as_ref().map(audit_actions).unwrap_or_default();

        PushResult {
            parsed,
            response,
            actions,
            reply_text,
            multimodal,
            audit,
        }
    }
}

/// Whether a `response.json` value carries a non-empty `media` array.
fn has_media(response: &Value) -> bool {
    response
        .get("media")
        .and_then(Value::as_array)
        .map(|m| !m.is_empty())
        .unwrap_or(false)
}

/// Choose the reply text to show the user from Call 2's output.
///
/// Prefers `response.json.text`. If the structured document is missing (e.g. the
/// model returned a bare object or non-array JSON), it makes one best-effort
/// parse of the cleaned reply before falling back to the raw text, so the UI is
/// never left blank.
fn reply_text_from(response: Option<&Value>, cleaned: &str) -> String {
    // Preferred: the structured response.json text field.
    if let Some(text) = response.and_then(|r| r.get("text")).and_then(Value::as_str) {
        return text.trim().to_string();
    }

    // Fallback: tolerate a bare `{ "text": ... }` object or an array the
    // document extractor declined (it requires a leading `[`).
    if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
        if let Some(text) = value
            .as_array()
            .and_then(|a| a.first())
            .and_then(|first| first.get("text"))
            .and_then(Value::as_str)
        {
            return text.trim().to_string();
        }
        if let Some(text) = value.get("text").and_then(Value::as_str) {
            return text.trim().to_string();
        }
    }

    // Not parseable JSON at all: show the raw text rather than nothing.
    cleaned.to_string()
}

/// Strip a leading/trailing Markdown code fence (```/```json) if present.
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // Drop an optional language tag on the opening fence's line.
    let after_lang = match after_open.split_once('\n') {
        Some((_lang, rest)) => rest,
        None => after_open,
    };
    after_lang
        .trim_end()
        .strip_suffix("```")
        .unwrap_or(after_lang)
        .trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_contacts_always_leads_with_the_default() {
        let contacts = parse_contacts(
            "Jonty",
            "+27725697683",
            "architect of the gaia brain design",
            "",
        );
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].name, "Jonty");
        assert_eq!(contacts[0].phone, "+27725697683");
        assert_eq!(contacts[0].role, "architect of the gaia brain design");
    }

    #[test]
    fn parse_contacts_appends_and_skips_blank_extra_entries() {
        let contacts = parse_contacts(
            "Jonty",
            "+27725697683",
            "architect",
            "Blinky|+27000000000|close friend; ; Mom|+27111111111|family",
        );
        assert_eq!(contacts.len(), 3);
        assert_eq!(contacts[1].name, "Blinky");
        assert_eq!(contacts[2].name, "Mom");
        assert_eq!(contacts[2].phone, "+27111111111");
    }

    #[test]
    fn canonical_store_maps_names_and_aliases() {
        assert_eq!(canonical_store("gaiaconnections"), Some("GaiaConnections"));
        assert_eq!(canonical_store("connections"), Some("GaiaConnections"));
        assert_eq!(canonical_store("gaiadatalake"), Some("GaiaDataLake"));
        assert_eq!(canonical_store("data lake"), Some("GaiaDataLake"));
        assert_eq!(canonical_store("gaiadiary"), Some("GaiaDiary"));
        assert_eq!(canonical_store("gaiakb"), Some("GaiaKB"));
        assert_eq!(canonical_store("knowledge"), Some("GaiaKB"));
        assert_eq!(canonical_store("web"), None);
    }

    #[test]
    fn read_urgency_accepts_numbers_and_strings() {
        assert_eq!(read_urgency(&json!({ "urgency": 0.5 })), Some(0.5));
        assert_eq!(read_urgency(&json!({ "urgency": "0.7" })), Some(0.7));
        assert_eq!(read_urgency(&json!({ "urgency": "nope" })), None);
        assert_eq!(read_urgency(&json!({})), None);
    }

    #[test]
    fn urgency_delivery_threshold_gates_above_one_third() {
        // A record is delivered only when its urgency strictly exceeds 0.33.
        let at_threshold = json!({ "target": "Push", "message": "x", "urgency": 0.33 });
        let above = json!({ "target": "Push", "message": "x", "urgency": 0.34 });
        assert!(!audit_actions(&json!({ "actions": [at_threshold] })).push[0].delivered);
        assert!(audit_actions(&json!({ "actions": [above] })).push[0].delivered);
    }

    #[test]
    fn audit_classifies_every_required_record() {
        let actions = json!({
            "version": "1.0",
            "actions": [
                { "id": "a1", "kind": "send", "target": "WhatsApp",
                  "to_name": "Jonty", "to_phone": "+27725697683",
                  "message": "Hi", "urgency": 0.7 },
                { "id": "a2", "kind": "send", "target": "Push",
                  "message": "ping", "urgency": 0.2 },
                { "id": "a3", "kind": "actuate", "target": "Edwino",
                  "instruction": { "robot": "Edwino", "face": "happy" } },
                { "id": "a4", "kind": "upsert", "target": "GaiaConnections", "payload": {} },
                { "id": "a5", "kind": "upsert", "target": "GaiaKB", "payload": {} },
                { "id": "a6", "kind": "upsert", "target": "GaiaDataLake", "payload": {} },
                { "id": "a7", "kind": "upsert", "target": "GaiaDiary", "payload": {} }
            ]
        });

        let audit = audit_actions(&actions);
        assert_eq!(audit.whatsapp.len(), 1);
        assert!(audit.whatsapp[0].delivered); // 0.7 > 0.33
        assert_eq!(audit.push.len(), 1);
        assert!(!audit.push[0].delivered); // 0.2 <= 0.33
        assert_eq!(audit.actuate.len(), 1);
        assert_eq!(audit.store_writes.len(), 4);
        assert!(audit.missing_stores().is_empty());
    }

    #[test]
    fn audit_reports_missing_stores() {
        let actions = json!({
            "actions": [
                { "id": "a5", "kind": "upsert", "target": "GaiaKB", "payload": {} }
            ]
        });
        let audit = audit_actions(&actions);
        assert_eq!(audit.store_writes.len(), 1);
        assert_eq!(
            audit.missing_stores(),
            vec!["GaiaConnections", "GaiaDataLake", "GaiaDiary"]
        );
    }

    #[test]
    fn summarize_actions_lists_every_side_effect() {
        let actions = json!({
            "actions": [
                { "id": "a1", "kind": "send", "target": "WhatsApp",
                  "to_name": "Jonty", "message": "Hi", "urgency": 0.7 },
                { "id": "a2", "kind": "send", "target": "Push",
                  "message": "ping", "urgency": 0.2 },
                { "id": "a3", "kind": "actuate", "target": "Edwino",
                  "instruction": { "movement": { "drive": "forward" }, "face": "happy" } },
                { "id": "a4", "kind": "upsert", "target": "GaiaKB", "payload": {} }
            ]
        });

        let summary = summarize_actions(&audit_actions(&actions));
        assert!(summary.contains("WhatsApp to Jonty: sent (urgency 0.70)"));
        assert!(summary.contains("Push: held (urgency 0.20)"));
        assert!(summary.contains("Edwino: drive forward, face happy"));
        assert!(summary.contains("Saved to: GaiaKB"));
    }

    #[test]
    fn summarize_actions_is_empty_without_side_effects() {
        assert_eq!(summarize_actions(&ActionAudit::default()), "");
    }

    #[test]
    fn describe_actuate_falls_back_when_unstructured() {
        assert_eq!(describe_actuate(&json!({})), "actuate");
    }

    #[test]
    fn build_execution_prompt_embeds_contacts_question_and_context() {
        let contacts = vec![WhatsAppContact {
            name: "Jonty".to_string(),
            phone: "+27725697683".to_string(),
            role: "architect".to_string(),
        }];
        let prompt = build_execution_prompt(
            "threadkeeper",
            "What do you know about me?",
            "## DataLakeResults\nsome context",
            &contacts,
            "2026-06-21T00:00:00Z",
        );

        // System carries the contact, the required-record rules, and identity.
        assert!(prompt.system.contains("+27725697683"));
        assert!(prompt.system.contains("WhatsApp"));
        assert!(prompt.system.contains("Edwino"));
        assert!(prompt.system.contains("GaiaConnections"));
        // User carries the question, the time, and the grounding context.
        assert!(prompt.user.contains("What do you know about me?"));
        assert!(prompt.user.contains("2026-06-21T00:00:00Z"));
        assert!(prompt.user.contains("some context"));
    }

    #[test]
    fn build_execution_prompt_handles_empty_context_and_contacts() {
        let prompt = build_execution_prompt("u", "q", "   ", &[], "t");
        assert!(prompt.system.contains("(no contacts configured)"));
        assert!(prompt
            .user
            .contains("(no research results were assembled this turn)"));
    }

    #[test]
    fn build_prompt_uses_the_controllers_contacts() {
        let controller = PushDataController::new(vec![WhatsAppContact {
            name: "Mara".to_string(),
            phone: "+27123456789".to_string(),
            role: "friend".to_string(),
        }]);
        let prompt = controller.build_prompt("u", "q", "ctx", "t");
        assert!(prompt.system.contains("Mara"));
        assert!(prompt.system.contains("+27123456789"));
    }

    #[test]
    fn process_extracts_text_from_response_array() {
        let raw = r#"[{"text":"Hi there","emote":"warm","medium":"console"},{"version":"1.0"}]"#;
        let result = PushDataController::process(raw);
        assert!(result.parsed);
        assert_eq!(result.reply_text, "Hi there");
        assert_eq!(result.response_text(), "Hi there");
    }

    #[test]
    fn process_extracts_text_from_bare_object() {
        // A bare response.json object (no array) still yields the reply text via
        // the fallback, even though no document array could be extracted.
        let result = PushDataController::process(r#"{"text":"hello"}"#);
        assert!(!result.parsed);
        assert_eq!(result.reply_text, "hello");
    }

    #[test]
    fn process_strips_code_fences_before_parsing() {
        let raw = "```json\n[{\"text\":\"fenced\"}]\n```";
        let result = PushDataController::process(raw);
        assert!(result.parsed);
        assert_eq!(result.reply_text, "fenced");
    }

    #[test]
    fn process_falls_back_to_raw_text_when_not_json() {
        let result = PushDataController::process("just words");
        assert!(!result.parsed);
        assert_eq!(result.reply_text, "just words");
    }

    #[test]
    fn process_detects_multimodal_media() {
        let raw = r#"[
          { "text": "look", "media": [ { "type": "image", "description": "a cat" } ] },
          { "actions": [] }
        ]"#;
        let result = PushDataController::process(raw);
        assert!(result.multimodal);
    }

    #[test]
    fn process_audits_actions_and_summarizes() {
        // A well-formed [response.json, actions.json] reply yields an audit and a
        // summary that names each planned side effect.
        let raw = r#"[
          { "text": "Hello" },
          { "actions": [
            { "id": "a1", "kind": "send", "target": "WhatsApp",
              "to_name": "Jonty", "message": "Hi", "urgency": 0.7 },
            { "id": "a4", "kind": "upsert", "target": "GaiaDiary", "payload": {} }
          ] }
        ]"#;
        let result = PushDataController::process(raw);
        assert_eq!(result.audit.whatsapp.len(), 1);
        let summary = result.actions_summary().expect("a summary");
        assert!(summary.contains("WhatsApp to Jonty: sent"));
        assert!(summary.contains("Saved to: GaiaDiary"));
    }

    #[test]
    fn process_returns_none_summary_without_actions() {
        // Only response.json present (no second element): nothing to summarize.
        assert!(PushDataController::process(r#"[{ "text": "Hi" }]"#)
            .actions_summary()
            .is_none());
        // Not parseable as a JSON array at all.
        assert!(PushDataController::process("just words")
            .actions_summary()
            .is_none());
    }
}
