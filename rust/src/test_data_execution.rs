//! The [`DataExecutionProbe`] type: an end-to-end self-test of Gaia's push pass.
//!
//! This module powers the `gaia-robot test-data-execution` subcommand (and the
//! `infra/DataExecution.ps1` wrapper). It is the **push-pass** counterpart to
//! [`crate::test_data_retrieval`]: where the retrieval probe exercises LLM Call
//! 1 (decide what to fetch and run the reads), this probe exercises **LLM Call
//! 2** — answer the user and emit the side-effecting `actions.json`.
//!
//! All of the actual push-pass logic — the Call 2 prompt, the reply parsing, and
//! the side-effect audit — lives in the shared [`crate::push_data_controller`],
//! which the **live cloud app** ([`crate::engine::Engine`]) drives too. This
//! module is therefore only the *test harness* around that controller: it reads
//! each captured turn, runs the controller, scores the result, and writes the
//! artifacts and summary. That split is the whole point — the probe validates
//! the exact code that runs in production.
//!
//! For each turn already captured under `tests/LLM1/t{N}/`, the probe:
//!
//! 1. Reads that turn's `responsedatacontext.md` (the deterministic grounding
//!    context assembled between Call 1 and Call 2) and the user's question.
//! 2. Asks the controller to build the Call 2 prompt and runs the model.
//! 3. Asks the controller to parse and **audit** the emitted `actions.json`,
//!    then checks that every required record was produced this turn: a WhatsApp
//!    message, a Push message, an Edwino actuate instruction, and an upsert into
//!    each of the four data stores (GaiaConnections, GaiaKB, GaiaDataLake,
//!    GaiaDiary).
//!
//! The probe is **read-only against production**: it validates the `actions.json`
//! that Call 2 *plans*; it never executes the writes/sends against live Cosmos,
//! WhatsApp, Push, or the robot. The full contract lives in
//! `tests/LLM2/DataExecutionSpec.md`.

use std::io::{self, Write};
use std::path::Path;

use serde_json::Value;

use crate::llm::LlmClient;
use crate::prompt::now_rfc3339;
use crate::push_data_controller::{urgency_in_range, ActionAudit, PushDataController};

/// The user every turn is scoped to. Matches the `threadkeeper` exports under
/// `migrations/` and the retrieval probe, so the context lines up with real
/// seeded data.
const PROBE_USER_ID: &str = "threadkeeper";

/// The number of turns the probe runs, one per `tests/LLM1/t{N}/` folder.
const TURN_COUNT: usize = 6;

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
    ///
    /// Creates the directory if it does not exist, then removes any existing
    /// `.json` files in that folder so stale artifacts from earlier runs do not
    /// leak into the new output.
    fn write_to(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        clear_json_files_in_dir(dir)?;

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

/// Delete every `.json` file directly inside `dir`.
///
/// Used to wipe stale artifacts from a previous run before the current turn's
/// files are written, so leftover files (e.g. a `response.json` from a run that
/// has since stopped producing one) do not linger in `tests/LLM2/t{N}`.
fn clear_json_files_in_dir(dir: &Path) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let is_json = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("json"))
            .unwrap_or(false);
        if is_json {
            std::fs::remove_file(path)?;
        }
    }

    Ok(())
}

/// Runs the data-execution self-test end to end.
///
/// Holds the live model client (always required) and the shared
/// [`PushDataController`] (which owns the contact list, the Call 2 prompt, and
/// the reply audit). Built with [`DataExecutionProbe::from_env`], driven with
/// [`DataExecutionProbe::run`].
pub struct DataExecutionProbe {
    /// The chat model client used for LLM Call 2.
    llm: LlmClient,
    /// The shared push-pass controller — the exact code the cloud app runs.
    controller: PushDataController,
}

impl DataExecutionProbe {
    /// Build a probe from the process environment.
    ///
    /// A model is mandatory (the probe cannot run Call 2 without one); the
    /// controller resolves the contact list from `GAIA_WHATSAPP_*` (with the
    /// documented `Jonty` default). Returns a clear error string when the model
    /// is missing or misconfigured, so the subcommand can print it and exit
    /// non-zero.
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
            controller: PushDataController::from_env(),
        })
    }

    /// Run the probe over the captured turns, write a report, and return whether
    /// the self-test passed.
    ///
    /// `input_dir` is where each turn's `responsedatacontext.md` is read from
    /// (the `tests/LLM1` folder). When `output_dir` is set, per-turn artifacts
    /// are written into `output_dir/t1 … t6`. When `only` is `Some(n)`, run only
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
        writeln!(out, "  contacts: {}", self.controller.contacts().len())?;
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
    ///
    /// The prompt building and the reply audit are delegated to the shared
    /// [`PushDataController`]; this method only orchestrates and scores.
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
        // The controller owns the prompt; the probe just runs the model.
        let requested_at = now_rfc3339();
        let prompt =
            self.controller
                .build_prompt(PROBE_USER_ID, &question, &context, &requested_at);
        let reply = match self.llm.complete(&prompt.system, &prompt.user) {
            // Self-tests only need the reply text, not the reported model.
            Ok(reply) => reply.content,
            Err(err) => {
                metrics.notes.push(format!("LLM Call 2 failed: {err}"));
                return (metrics, artifacts);
            }
        };
        metrics.llm_ok = true;
        artifacts.raw_reply = Some(reply.clone());

        // --- Process the reply through the shared push controller --------
        let push = PushDataController::process(&reply);
        artifacts.response = push.response.clone();
        artifacts.actions = push.actions.clone();

        if !push.parsed {
            metrics
                .notes
                .push("could not parse the JSON array from the reply".to_string());
            return (metrics, artifacts);
        }

        // --- Validate response.json --------------------------------------
        if push.response.is_some() {
            let text = push.response_text();
            metrics.response_ok = !text.is_empty();
            metrics.multimodal = push.multimodal;
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
        if push.actions.is_none() {
            metrics
                .notes
                .push("no actions.json (second array element)".to_string());
            return (metrics, artifacts);
        }
        let audit = &push.audit;

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

        metrics.success = metrics.llm_ok
            && metrics.response_ok
            && metrics.whatsapp_ok
            && metrics.push_ok
            && metrics.actuate_ok
            && missing.is_empty();

        artifacts.audit = push.audit.clone();

        (metrics, artifacts)
    }
}

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
    fn truncate_appends_ellipsis_only_when_shortened() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 5), "abcde…");
    }

    /// A `TurnMetrics` fixture with every flag set as requested.
    fn metrics(folder: &str, success: bool) -> TurnMetrics {
        TurnMetrics {
            folder: folder.to_string(),
            question: "What do you know about hiking trails near the coast?".to_string(),
            llm_ok: true,
            response_ok: true,
            whatsapp_ok: true,
            push_ok: success,
            actuate_ok: success,
            stores_covered: if success { 4 } else { 2 },
            multimodal: false,
            success,
            notes: Vec::new(),
        }
    }

    /// Create a unique, empty temp directory for a file-IO test.
    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("gaia_exec_{tag}_{}_{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn yes_no_maps_booleans_to_words() {
        assert_eq!(yes_no(true), "yes");
        assert_eq!(yes_no(false), "no");
    }

    #[test]
    fn pretty_pretty_prints_json_and_pretty_or_raw_falls_back() {
        let value = serde_json::json!({"a": 1});
        assert!(pretty(&value).contains("\"a\": 1"));

        // A raw reply with no Call-1 array passes through unchanged.
        assert_eq!(pretty_or_raw("just text"), "just text");
        // A reply that embeds a JSON array is pretty-printed.
        let pretty_array = pretty_or_raw("prefix [ {\"id\":\"q1\"} ] suffix");
        assert!(pretty_array.contains("\"id\": \"q1\""));
    }

    #[test]
    fn format_metrics_table_has_a_header_and_one_row_per_turn() {
        let table = format_metrics_table(&[metrics("t1", true), metrics("t2", false)]);
        assert!(table.contains("Turn  WhatsApp  Push  Actuate  Stores  Multimodal  Result"));
        assert!(table.contains("t1"));
        assert!(table.contains("PASS"));
        assert!(table.contains("t2"));
        assert!(table.contains("FAIL"));
        // The stores column renders as "n/4".
        assert!(table.contains("4/4"));
    }

    #[test]
    fn write_to_writes_every_artifact_and_clears_stale_json() {
        let dir = unique_temp_dir("write_to");
        // A stale file from a previous run must be removed first.
        std::fs::write(dir.join("stale.json"), "{}").expect("seed stale file");

        let artifacts = TurnArtifacts {
            raw_reply: Some("[ {\"id\":\"q1\"} ]".to_string()),
            response: Some(serde_json::json!({"text": "hi"})),
            actions: Some(serde_json::json!([{"id": "q1"}])),
            audit: ActionAudit::default(),
        };
        artifacts.write_to(&dir).expect("write artifacts");

        // The fixed side-effect files plus the optional documents all exist.
        for name in [
            "reply.json",
            "response.json",
            "actions.json",
            "whatsapp.json",
            "push.json",
            "actuate.json",
            "writes.json",
        ] {
            assert!(dir.join(name).exists(), "missing {name}");
        }
        // The stale file was cleared.
        assert!(!dir.join("stale.json").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_artifacts_write_only_the_fixed_side_effect_files() {
        let dir = unique_temp_dir("empty");
        TurnArtifacts::empty()
            .write_to(&dir)
            .expect("write empties");

        // The optional documents are absent when their sources are `None`.
        assert!(!dir.join("reply.json").exists());
        assert!(!dir.join("response.json").exists());
        assert!(!dir.join("actions.json").exists());
        // The four side-effect files are always written.
        assert!(dir.join("whatsapp.json").exists());
        assert!(dir.join("writes.json").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_summary_md_renders_a_markdown_table() {
        let dir = unique_temp_dir("summary");
        write_summary_md(&dir, &[metrics("t1", true), metrics("t2", false)], false)
            .expect("write summary");

        let md = std::fs::read_to_string(dir.join("TestSummary.md")).expect("read summary");
        assert!(md.contains("# Data-Execution Self-Test Summary"));
        assert!(md.contains("FAIL ❌"));
        assert!(md.contains("| 1 |"));
        assert!(md.contains("| 2 |"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
