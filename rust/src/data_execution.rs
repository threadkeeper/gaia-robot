//! The [`DataExecutionProbe`] type: an end-to-end self-test of Gaia's push pass.
//!
//! This module powers the `gaia-robot test-data-execution` subcommand (and the
//! `infra/DataExecution.ps1` wrapper). It is the **push-pass** counterpart to
//! [`crate::test_data_retrieval`]: where the retrieval probe exercises LLM Call
//! 1 (decide what to fetch and run the reads), this probe exercises **LLM Call
//! 2** — answer the user and emit the side-effecting `actions.json`.
//!
//! For each turn already captured under `tests/LLM1/t{N}/`, the probe:
//!
//! 1. Reads that turn's `responsedatacontext.md` (the deterministic grounding
//!    context assembled between Call 1 and Call 2) and the user's question.
//! 2. Builds a focused Call 2 prompt that hands Gaia the question, the context,
//!    and her list of WhatsApp contacts, then asks her to emit two documents:
//!    `response.json` (a possibly multi-modal reply) and `actions.json` (the
//!    side effects to carry out).
//! 3. Parses and **audits** the emitted `actions.json`, checking that every
//!    required record was produced this turn: a WhatsApp message, a Push
//!    message, an Edwino actuate instruction, and an upsert into each of the
//!    four data stores (GaiaConnections, GaiaKB, GaiaDataLake, GaiaDiary).
//!
//! The probe is **read-only against production**: it validates the `actions.json`
//! that Call 2 *plans*; it never executes the writes/sends against live Cosmos,
//! WhatsApp, Push, or the robot. The full contract lives in
//! `tests/LLM2/DataExecutionSpec.md`.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;

use serde_json::Value;

use crate::llm::{value_from_env, LlmClient};
use crate::prompt::now_rfc3339;

/// The user every turn is scoped to. Matches the `threadkeeper` exports under
/// `migrations/` and the retrieval probe, so the context lines up with real
/// seeded data.
const PROBE_USER_ID: &str = "threadkeeper";

/// The number of turns the probe runs, one per `tests/LLM1/t{N}/` folder.
const TURN_COUNT: usize = 5;

/// The four data stores Gaia must write to every turn.
const REQUIRED_STORES: [&str; 4] = ["GaiaConnections", "GaiaKB", "GaiaDataLake", "GaiaDiary"];

/// The WhatsApp/Push urgency above which the delivery API actually executes a
/// message. Records at or below this are still produced, just suppressed.
const URGENCY_DELIVERY_THRESHOLD: f64 = 0.33;

/// Fallback default contact, used when the `GAIA_WHATSAPP_DEFAULT_*` env vars
/// are absent. Mirrors the documented default in `infra/.env.sample`.
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
    fn missing_stores(&self) -> Vec<&'static str> {
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
fn urgency_in_range(urgency: f64) -> bool {
    (0.0..=1.0).contains(&urgency)
}

/// Per-turn outcome of the push-pass probe.
#[derive(Debug, Clone)]
pub struct TurnMetrics {
    /// The turn folder name, e.g. `t1`.
    pub folder: String,
    /// The user's question for this turn.
    pub question: String,
    /// Whether LLM Call 2 returned a usable reply.
    pub llm_ok: bool,
    /// Whether `response.json` parsed with non-empty text.
    pub response_ok: bool,
    /// Whether a valid WhatsApp record was produced.
    pub whatsapp_ok: bool,
    /// Whether a valid Push record was produced.
    pub push_ok: bool,
    /// Whether a valid Edwino actuate record was produced.
    pub actuate_ok: bool,
    /// How many of the four stores received an upsert this turn.
    pub stores_covered: usize,
    /// Whether the reply carried multi-modal media.
    pub multimodal: bool,
    /// Overall pass/fail for this turn.
    pub success: bool,
    /// Human-readable notes explaining any failure.
    pub notes: Vec<String>,
}

/// Raw artifacts written to disk for one turn.
struct TurnArtifacts {
    /// The raw LLM Call 2 reply, before parsing.
    raw_reply: Option<String>,
    /// The parsed `response.json` document.
    response: Option<Value>,
    /// The parsed `actions.json` document.
    actions: Option<Value>,
    /// The audited side-effect records.
    audit: ActionAudit,
}

impl TurnArtifacts {
    fn empty() -> Self {
        Self {
            raw_reply: None,
            response: None,
            actions: None,
            audit: ActionAudit::default(),
        }
    }

    /// Write the artifacts as pretty-printed JSON files into `dir`.
    fn write_to(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;

        if let Some(reply) = &self.raw_reply {
            std::fs::write(dir.join("reply.json"), pretty_or_raw(reply))?;
        }
        if let Some(response) = &self.response {
            std::fs::write(dir.join("response.json"), pretty(response))?;
        }
        if let Some(actions) = &self.actions {
            std::fs::write(dir.join("actions.json"), pretty(actions))?;
        }

        std::fs::write(
            dir.join("whatsapp.json"),
            serde_json::to_string_pretty(&self.audit.whatsapp).unwrap_or_default(),
        )?;
        std::fs::write(
            dir.join("push.json"),
            serde_json::to_string_pretty(&self.audit.push).unwrap_or_default(),
        )?;
        std::fs::write(
            dir.join("actuate.json"),
            serde_json::to_string_pretty(&self.audit.actuate).unwrap_or_default(),
        )?;
        std::fs::write(
            dir.join("writes.json"),
            serde_json::to_string_pretty(&self.audit.store_actions).unwrap_or_default(),
        )?;

        Ok(())
    }
}

/// Runs the data-execution self-test end to end.
///
/// Holds the live model client (always required) and the resolved WhatsApp
/// contact list. Built with [`DataExecutionProbe::from_env`], driven with
/// [`DataExecutionProbe::run`].
pub struct DataExecutionProbe {
    /// The chat model client used for LLM Call 2.
    llm: LlmClient,
    /// The contacts Gaia may WhatsApp.
    contacts: Vec<WhatsAppContact>,
}

impl DataExecutionProbe {
    /// Build a probe from the process environment.
    ///
    /// A model is mandatory (the probe cannot run Call 2 without one); the
    /// contact list is resolved from `GAIA_WHATSAPP_*` (with the documented
    /// `Jonty` default). Returns a clear error string when the model is missing
    /// or misconfigured, so the subcommand can print it and exit non-zero.
    pub fn from_env() -> Result<Self, String> {
        let llm = match LlmClient::from_env() {
            Ok(Some(client)) => client,
            Ok(None) => {
                return Err(
                    "LLM is not enabled. Set GAIA_MODE=dev (or local) and configure a model \
                     (FOUNDRY_ENDPOINT + MODEL_ROUTER_DEPLOYMENT + FOUNDRY_API_KEY, or GITHUB_TOKEN)."
                        .to_string(),
                )
            }
            Err(err) => return Err(format!("LLM configuration error: {err}")),
        };

        Ok(Self {
            llm,
            contacts: contacts_from_env(),
        })
    }

    /// Run the probe over the captured turns, write a report, and return whether
    /// the self-test passed.
    ///
    /// `input_dir` is where each turn's `responsedatacontext.md` is read from
    /// (the `tests/LLM1` folder). When `output_dir` is set, per-turn artifacts
    /// are written into `output_dir/t1 … t5`. When `only` is `Some(n)`, run only
    /// turn `n` (1-based). The boolean is the gate: `true` only when every
    /// executed turn passed.
    pub fn run(
        &self,
        only: Option<usize>,
        input_dir: &Path,
        output_dir: Option<&Path>,
        out: &mut impl Write,
    ) -> io::Result<bool> {
        writeln!(
            out,
            "Gaia data-execution self-test (LLM Call 2 / push pass)"
        )?;
        writeln!(out, "  user_id : {PROBE_USER_ID}")?;
        writeln!(
            out,
            "  model   : {} ({})",
            self.llm.model(),
            self.llm.endpoint()
        )?;
        writeln!(out, "  contacts: {}", self.contacts.len())?;
        writeln!(out)?;

        // Decide which turns to run.
        let turns: Vec<usize> = match only {
            Some(n) if (1..=TURN_COUNT).contains(&n) => vec![n],
            Some(n) => {
                writeln!(out, "ERROR: turn {n} does not exist (1–{TURN_COUNT}).")?;
                return Ok(false);
            }
            None => (1..=TURN_COUNT).collect(),
        };

        let mut all = Vec::with_capacity(turns.len());
        for n in turns {
            let folder = format!("t{n}");
            writeln!(out, "[{n}/{TURN_COUNT}] {folder}")?;

            let (metrics, artifacts) = self.probe_one(input_dir, &folder);

            if let Some(dir) = output_dir {
                let q_dir = dir.join(&folder);
                if let Err(err) = artifacts.write_to(&q_dir) {
                    writeln!(
                        out,
                        "      - warning: could not write artifacts to {}: {err}",
                        q_dir.display()
                    )?;
                }
            }

            for note in &metrics.notes {
                writeln!(out, "      - {note}")?;
            }
            writeln!(
                out,
                "      => {} | WhatsApp {} | Push {} | Actuate {} | stores {}/4{}",
                if metrics.success { "PASS" } else { "FAIL" },
                if metrics.whatsapp_ok { "ok" } else { "—" },
                if metrics.push_ok { "ok" } else { "—" },
                if metrics.actuate_ok { "ok" } else { "—" },
                metrics.stores_covered,
                if metrics.multimodal {
                    " | multimodal"
                } else {
                    ""
                },
            )?;
            all.push(metrics);
        }

        writeln!(out)?;
        write!(out, "{}", format_metrics_table(&all))?;
        let pass = !all.is_empty() && all.iter().all(|m| m.success);
        writeln!(out)?;
        writeln!(out, "OVERALL: {}", if pass { "PASS" } else { "FAIL" })?;

        if let Some(dir) = output_dir {
            if let Err(err) = write_summary_md(dir, &all, pass) {
                writeln!(out, "warning: could not write TestSummary.md: {err}")?;
            }
        }

        Ok(pass)
    }

    /// Run one turn: read its context, call the model, audit the result.
    fn probe_one(&self, input_dir: &Path, folder: &str) -> (TurnMetrics, TurnArtifacts) {
        let mut metrics = TurnMetrics {
            folder: folder.to_string(),
            question: String::new(),
            llm_ok: false,
            response_ok: false,
            whatsapp_ok: false,
            push_ok: false,
            actuate_ok: false,
            stores_covered: 0,
            multimodal: false,
            success: false,
            notes: Vec::new(),
        };
        let mut artifacts = TurnArtifacts::empty();

        // --- Read this turn's grounding context --------------------------
        let context_path = input_dir.join(folder).join("responsedatacontext.md");
        let context = match std::fs::read_to_string(&context_path) {
            Ok(text) => text,
            Err(err) => {
                metrics
                    .notes
                    .push(format!("could not read {}: {err}", context_path.display()));
                return (metrics, artifacts);
            }
        };
        let question = parse_question(&context);
        metrics.question = question.clone();

        // --- LLM Call 2: answer + plan side effects ----------------------
        let requested_at = now_rfc3339();
        let prompt = build_execution_prompt(
            PROBE_USER_ID,
            &question,
            &context,
            &self.contacts,
            &requested_at,
        );
        let reply = match self.llm.complete(&prompt.system, &prompt.user) {
            Ok(reply) => reply,
            Err(err) => {
                metrics.notes.push(format!("LLM Call 2 failed: {err}"));
                return (metrics, artifacts);
            }
        };
        metrics.llm_ok = true;
        artifacts.raw_reply = Some(reply.clone());

        // --- Parse the two documents (response.json, actions.json) -------
        let documents = match crate::actions::extract_call1_array(&reply) {
            Some(docs) => docs,
            None => {
                metrics
                    .notes
                    .push("could not parse the JSON array from the reply".to_string());
                return (metrics, artifacts);
            }
        };
        let response = documents.first().cloned();
        let actions = documents.get(1).cloned();
        artifacts.response = response.clone();
        artifacts.actions = actions.clone();

        // --- Validate response.json --------------------------------------
        if let Some(response) = &response {
            let text = string_field(response, "text");
            metrics.response_ok = !text.is_empty();
            metrics.multimodal = response
                .get("media")
                .and_then(Value::as_array)
                .map(|m| !m.is_empty())
                .unwrap_or(false);
            if !metrics.response_ok {
                metrics
                    .notes
                    .push("response.json has no non-empty `text`".to_string());
            }
        } else {
            metrics
                .notes
                .push("no response.json (first array element)".to_string());
        }

        // --- Audit actions.json ------------------------------------------
        let Some(actions) = &actions else {
            metrics
                .notes
                .push("no actions.json (second array element)".to_string());
            return (metrics, artifacts);
        };
        let audit = audit_actions(actions);

        // WhatsApp: at least one with a phone, a message, and a valid urgency.
        metrics.whatsapp_ok = audit.whatsapp.iter().any(|m| {
            !m.to_phone.is_empty() && !m.message.is_empty() && urgency_in_range(m.urgency)
        });
        if !metrics.whatsapp_ok {
            metrics.notes.push(
                "missing a valid WhatsApp record (phone + message + urgency 0–1)".to_string(),
            );
        }

        // Push: at least one with a message and a valid urgency.
        metrics.push_ok = audit
            .push
            .iter()
            .any(|m| !m.message.is_empty() && urgency_in_range(m.urgency));
        if !metrics.push_ok {
            metrics
                .notes
                .push("missing a valid Push record (message + urgency 0–1)".to_string());
        }

        // Actuate: at least one Edwino instruction object.
        metrics.actuate_ok = audit.actuate.iter().any(Value::is_object);
        if !metrics.actuate_ok {
            metrics
                .notes
                .push("missing a valid Edwino actuate instruction".to_string());
        }

        // Store write-backs: all four required containers.
        metrics.stores_covered = audit.store_writes.len();
        let missing = audit.missing_stores();
        if !missing.is_empty() {
            metrics
                .notes
                .push(format!("missing store write-backs: {}", missing.join(", ")));
        }

        artifacts.audit = audit;

        metrics.success = metrics.llm_ok
            && metrics.response_ok
            && metrics.whatsapp_ok
            && metrics.push_ok
            && metrics.actuate_ok
            && missing.is_empty();

        (metrics, artifacts)
    }
}

/// The two chat messages sent to LLM Call 2 for the push-pass probe.
pub struct ExecutionPrompt {
    /// The system message: identity, document spec, contacts, output rule.
    pub system: String,
    /// The user message: this turn's question, time, and grounding context.
    pub user: String,
}

/// Build the focused Call 2 prompt for one turn of the data-execution test.
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

/// The two-document contract for LLM Call 2 in the data-execution test.
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

/// Extract the user's question from a `responsedatacontext.md` body.
///
/// The Response Data Context builder writes a `- **question:** <text>` header
/// line; this returns the text after it, or an empty string when absent.
fn parse_question(context: &str) -> String {
    const MARKER: &str = "**question:**";
    for line in context.lines() {
        if let Some(idx) = line.find(MARKER) {
            return line[idx + MARKER.len()..].trim().to_string();
        }
    }
    String::new()
}

/// Pretty-print a JSON value, falling back to its compact form.
fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Pretty-print the JSON array embedded in a raw reply, or return it unchanged.
fn pretty_or_raw(reply: &str) -> String {
    if let Some(docs) = crate::actions::extract_call1_array(reply) {
        let array = Value::Array(docs);
        return pretty(&array);
    }
    reply.to_string()
}

/// Render the per-turn results as an aligned text table.
fn format_metrics_table(all: &[TurnMetrics]) -> String {
    let mut out = String::new();
    out.push_str("Turn  WhatsApp  Push  Actuate  Stores  Multimodal  Result\n");
    out.push_str("----  --------  ----  -------  ------  ----------  ------\n");
    for m in all {
        out.push_str(&format!(
            "{:<4}  {:<8}  {:<4}  {:<7}  {:<6}  {:<10}  {}\n",
            m.folder,
            yes_no(m.whatsapp_ok),
            yes_no(m.push_ok),
            yes_no(m.actuate_ok),
            format!("{}/4", m.stores_covered),
            yes_no(m.multimodal),
            if m.success { "PASS" } else { "FAIL" },
        ));
    }
    out
}

/// `yes`/`no` for a boolean column.
fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

/// Write a markdown summary table for the run, mirroring `tests/LLM1/TestSummary.md`.
fn write_summary_md(dir: &Path, all: &[TurnMetrics], pass: bool) -> io::Result<()> {
    let mut md = String::new();
    md.push_str("# Data-Execution Self-Test Summary\n\n");
    md.push_str(&format!(
        "**Overall: {}**\n\n",
        if pass { "PASS ✅" } else { "FAIL ❌" }
    ));
    md.push_str(
        "| # | Question | LLM | WhatsApp | Push | Actuate | Stores | Multimodal | Result |\n",
    );
    md.push_str(
        "|---|----------|-----|----------|------|---------|--------|------------|--------|\n",
    );
    for (i, m) in all.iter().enumerate() {
        let question = truncate(&m.question, 60);
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {}/4 | {} | {} |\n",
            i + 1,
            question,
            if m.llm_ok { "ok" } else { "fail" },
            yes_no(m.whatsapp_ok),
            yes_no(m.push_ok),
            yes_no(m.actuate_ok),
            m.stores_covered,
            yes_no(m.multimodal),
            if m.success { "PASS ✅" } else { "FAIL ❌" },
        ));
    }
    std::fs::write(dir.join("TestSummary.md"), md)
}

/// Truncate a string to `max` characters, appending `…` when shortened.
fn truncate(text: &str, max: usize) -> String {
    let trimmed: String = text.chars().take(max).collect();
    if trimmed.chars().count() < text.chars().count() {
        format!("{trimmed}…")
    } else {
        trimmed
    }
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
    fn parse_question_reads_the_marker_line() {
        let context = "# Response Data Context\n\n- **user_id:** threadkeeper\n\
                       - **question:** What do you know about hiking?\n- **requested_at:** now\n";
        assert_eq!(parse_question(context), "What do you know about hiking?");
    }

    #[test]
    fn parse_question_is_empty_when_absent() {
        assert_eq!(parse_question("no marker here"), "");
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
    fn truncate_appends_ellipsis_only_when_shortened() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 5), "abcde…");
    }
}
