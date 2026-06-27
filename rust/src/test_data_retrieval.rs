//! The [`DataRetrievalProbe`] type: an end-to-end self-test of Gaia's pull pass.
//!
//! This module powers the `gaia-robot test-data-retrieval` subcommand (and the
//! `infra/TestDataRetrieval.ps1` wrapper). Its job is to prove that the whole
//! *retrieval* half of a turn actually works against live infrastructure before
//! we ship a new build:
//!
//! 1. Ask the model **six** fixed questions of varying length and subject, each
//!    scoped to the `threadkeeper` user (user isolation).
//! 2. Parse the `actions.json` document out of each LLM Call 1 reply (the same
//!    [`crate::actions::parse_call1_actions`] parser the console pull pass uses).
//! 3. Execute every retrieval action: the Cosmos-backed queries through
//!    [`crate::executor::Executor`] and the `Web` action through the Brave
//!    [`crate::web_search::BraveClient`].
//! 4. Report, per question, a simple pass/fail plus **rows retrieved** and the
//!    **data size in KB** — for Cosmos, for Brave, and combined.
//!
//! The probe is deliberately strict: it only passes when the model answered,
//! actions parsed, and **every** retrieval it attempted succeeded. The
//! subcommand turns that boolean into the process exit code, so it can be run
//! on demand by a developer *and* used as a hard gate in CI: any failure exits
//! non-zero and halts the deploy.
//!
//! Configuration reuses the existing clients verbatim ([`LlmClient::from_env`],
//! [`CosmosClient::from_env`], [`BraveClient::from_env`]), so the probe needs
//! the same `GAIA_MODE=dev` setup as the rest of the dev/local code paths.

use std::io::{self, Write};
use std::path::Path;

use crate::actions::ActionsFile;
use crate::cosmos::CosmosClient;
use crate::embeddings::EmbeddingClient;
use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt};
use crate::pull_data_controller::{PullDataController, QueryAudit};
use crate::web_search::BraveClient;

/// The user every probe question is scoped to. Matches the `threadkeeper`
/// exports under `migrations/`, so the queries run against real seeded data.
const PROBE_USER_ID: &str = "threadkeeper";

/// The six fixed probe questions, chosen to vary in **length** and **subject**
/// so the model authors a spread of retrieval actions (personal recall, durable
/// facts, fresh web facts, and relationship/diary lookups).
const PROBE_QUESTIONS: [&str; 6] = [
    // 1. Double-barrelled: GaiaKB facts + GaiaDataLake conversation recall.
    "What do you know about Jonty's hobbies and interests and can you look up what you told \
     me about hiking recently?",
    // 2. Medium, personal recall (UsersDataLake / GaiaDataLake territory).
    "Remind me what we talked about regarding the robot's adventures in the forest recently.",
    // 3. Long, multi-topic personal synthesis (facts + history).
    "Can you summarise everything you know about my interests in music, books, and the \
     outdoors, note how any of those have shifted over the past month, and tie that back to \
     anything specific from our recent conversations?",
    // 4. Web search + GaiaKB opinion recall.
    "What are the latest developments in Mars exploration this year, and do you believe Mars \
     was once populated by people? Check your knowledge base for anything you know about Mars.",
    // 5. Relationship / diary lookup.
    "How has our friendship been going lately, and is there anything you noted in your diary \
     about me?",
    // 6. Fresh web facts: live weather, exercising the Brave web-search action.
    "Hi Gaia , please can you check what the temperature and the chance of rain is right now \
     in Cape Town South Africa",
];

/// Per-question retrieval metrics gathered by the probe.
///
/// Everything the requirement asks for lives here: a `success` flag, the number
/// of `rows` retrieved (Cosmos records + Brave results), and the data size in
/// KB. The Cosmos and Brave figures are kept separate as well so a failure is
/// easy to attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionMetrics {
    /// The question that was asked.
    pub question: String,
    /// Whether LLM Call 1 returned a usable reply.
    pub llm_ok: bool,
    /// How many actions parsed out of the `actions.json` document.
    pub actions_parsed: usize,
    /// How many Cosmos-backed (non-`Web`) actions were executed.
    pub cosmos_actions: usize,
    /// Total records returned across all Cosmos actions.
    pub cosmos_rows: usize,
    /// Total bytes of the Cosmos records returned (serialized JSON length).
    pub cosmos_bytes: usize,
    /// How many `Web` actions were executed against Brave.
    pub web_actions: usize,
    /// Total results returned across all Brave searches.
    pub web_rows: usize,
    /// Total bytes of the Brave results returned (serialized JSON length).
    pub web_bytes: usize,
    /// Per-container breakdown: (container_name, rows, bytes).
    pub container_details: Vec<(String, usize, usize)>,
    /// Overall pass/fail for this question (see [`DataRetrievalProbe::probe_one`]).
    pub success: bool,
    /// Human-readable notes explaining any failure or skipped retrieval.
    pub notes: Vec<String>,
}

impl QuestionMetrics {
    /// Total rows retrieved this question: Cosmos records plus Brave results.
    pub fn total_rows(&self) -> usize {
        self.cosmos_rows + self.web_rows
    }

    /// Total retrieved data size in kilobytes (Cosmos plus Brave).
    pub fn total_kb(&self) -> f64 {
        bytes_to_kb(self.cosmos_bytes + self.web_bytes)
    }

    /// Record rows and bytes for a specific container.
    fn record_container(&mut self, container: &str, rows: usize, bytes: usize) {
        if let Some(entry) = self
            .container_details
            .iter_mut()
            .find(|(c, _, _)| c == container)
        {
            entry.1 += rows;
            entry.2 += bytes;
        } else {
            self.container_details
                .push((container.to_string(), rows, bytes));
        }
    }
}

/// Raw artifacts collected during a single probe question, written to disk for
/// human review and debugging.
struct ProbeArtifacts {
    /// The raw LLM Call 1 reply, before parsing.
    raw_reply: Option<String>,
    /// The parsed actions.json document (element 0 of the Call 1 array).
    actions: Option<ActionsFile>,
    /// The exact Cosmos SQL that ran for each query action, so the artifact can
    /// be verified against what the model authored (keyword vs semantic).
    queries: Vec<QueryAudit>,
    /// Per-action retrieval results keyed by `(action_id, container)`.
    results: Vec<(String, String, Vec<serde_json::Value>)>,
    /// The Response Data Context the shared pull controller assembled this turn
    /// — the exact grounding document LLM Call 2 would receive in the cloud app.
    context: String,
}

impl ProbeArtifacts {
    fn empty() -> Self {
        Self {
            raw_reply: None,
            actions: None,
            queries: Vec::new(),
            results: Vec::new(),
            context: String::new(),
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
            let path = dir.join("reply.json");
            let pretty = pretty_print_reply(reply);
            std::fs::write(&path, pretty)?;
        }

        if let Some(actions) = &self.actions {
            let path = dir.join("actions.json");
            let json = serde_json::to_string_pretty(actions).unwrap_or_default();
            std::fs::write(&path, json)?;
        }

        if !self.queries.is_empty() {
            let path = dir.join("queries.json");
            let json = serde_json::to_string_pretty(&self.queries).unwrap_or_default();
            std::fs::write(&path, json)?;
        }

        for (action_id, container, records) in &self.results {
            let filename = format!("{action_id}_{container}.json");
            let path = dir.join(filename);
            let json = serde_json::to_string_pretty(records).unwrap_or_default();
            std::fs::write(&path, json)?;
        }

        // The Response Data Context was already assembled by the shared pull
        // controller (the same builder the cloud app uses); write it verbatim
        // alongside the JSON artifacts as `responsedatacontext.md`.
        std::fs::write(dir.join("responsedatacontext.md"), &self.context)?;

        Ok(())
    }
}

/// Delete every `.json` file directly inside `dir`.
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

/// Runs the data-retrieval self-test end to end.
///
/// Holds the live model client (always required) plus the optional Cosmos and
/// Brave clients. Built with [`DataRetrievalProbe::from_env`], driven with
/// [`DataRetrievalProbe::run`].
pub struct DataRetrievalProbe {
    /// The chat model client used for LLM Call 1. Always required.
    llm: LlmClient,
    /// The Cosmos client, or `None` when Cosmos is not configured.
    cosmos: Option<CosmosClient>,
    /// The embedding client used when actions choose semantic mode.
    embedder: Option<EmbeddingClient>,
    /// The Brave web-search client, or `None` when web search is not configured.
    web: Option<BraveClient>,
}

impl DataRetrievalProbe {
    /// Build a probe from the process environment.
    ///
    /// Reuses the exact same configuration the console app and server use. A
    /// model is mandatory (the probe cannot ask questions without one); Cosmos
    /// and Brave are resolved if configured. Returns a clear, human-readable
    /// error string when the model is missing or misconfigured, so the
    /// subcommand can print it and exit non-zero (a self-test that cannot run
    /// has validated nothing).
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

        let cosmos =
            CosmosClient::from_env().map_err(|err| format!("Cosmos configuration error: {err}"))?;
        let embedder = EmbeddingClient::from_env()
            .map_err(|err| format!("Embeddings configuration error: {err}"))?;
        let web = BraveClient::from_env();

        Ok(Self {
            llm,
            cosmos,
            embedder,
            web,
        })
    }

    /// Run probe questions, write a report to `out`, and return whether the
    /// self-test passed.
    ///
    /// When `only` is `Some(n)`, run only question `n` (1-based). When `None`,
    /// run all six. When `output_dir` is set, write pretty-printed JSON
    /// artifacts into `output_dir/q1/`, `output_dir/q2/`, etc. The boolean is
    /// the gate: `true` only when every executed question succeeded. The caller
    /// maps it to the process exit code.
    pub fn run(
        &self,
        only: Option<usize>,
        output_dir: Option<&Path>,
        out: &mut impl Write,
    ) -> io::Result<bool> {
        writeln!(out, "Gaia data-retrieval self-test")?;
        writeln!(out, "  user_id : {PROBE_USER_ID}")?;
        writeln!(
            out,
            "  model   : {} ({})",
            self.llm.model(),
            self.llm.endpoint()
        )?;
        writeln!(
            out,
            "  cosmos  : {}",
            match &self.cosmos {
                Some(client) => format!("enabled ({})", client.endpoint()),
                None => "DISABLED (set COSMOS_ENDPOINT + COSMOS_AAD_TOKEN)".to_string(),
            }
        )?;
        writeln!(
            out,
            "  embed   : {}",
            match &self.embedder {
                Some(client) => format!("enabled ({})", client.endpoint()),
                None => "DISABLED (set FOUNDRY_ENDPOINT + EMBEDDING_DEPLOYMENT + credential)"
                    .to_string(),
            }
        )?;
        writeln!(
            out,
            "  brave   : {}",
            match &self.web {
                Some(client) => format!("enabled ({})", client.endpoint()),
                None => "DISABLED (set BRAVE_SEARCH_API_KEY)".to_string(),
            }
        )?;
        writeln!(out)?;

        // Decide which questions to run.
        let questions: Vec<(usize, &str)> = match only {
            Some(n) if n >= 1 && n <= PROBE_QUESTIONS.len() => {
                vec![(n - 1, PROBE_QUESTIONS[n - 1])]
            }
            Some(n) => {
                writeln!(
                    out,
                    "ERROR: question {n} does not exist (1–{}).",
                    PROBE_QUESTIONS.len()
                )?;
                return Ok(false);
            }
            None => PROBE_QUESTIONS.iter().copied().enumerate().collect(),
        };

        // Probe each question in turn, collecting its metrics for the summary.
        let mut all = Vec::with_capacity(questions.len());
        for (index, question) in &questions {
            writeln!(out, "[{}/{}] {question}", index + 1, PROBE_QUESTIONS.len())?;
            let (metrics, artifacts) = self.probe_one(question);

            // Write artifacts to disk when an output directory is configured.
            if let Some(dir) = output_dir {
                let q_dir = dir.join(format!("t{}", index + 1));
                if let Err(err) = artifacts.write_to(&q_dir) {
                    writeln!(
                        out,
                        "      - warning: could not write artifacts to {}: {err}",
                        q_dir.display()
                    )?;
                }
            }

            // Echo the per-question outcome immediately so a long run shows progress.
            for note in &metrics.notes {
                writeln!(out, "      - {note}")?;
            }
            writeln!(
                out,
                "      => {} | rows {} | {:.2} KB",
                if metrics.success { "PASS" } else { "FAIL" },
                metrics.total_rows(),
                metrics.total_kb(),
            )?;
            all.push(metrics);
        }

        // Final summary table plus the overall verdict.
        writeln!(out)?;
        write!(out, "{}", format_metrics_table(&all))?;
        let pass = overall_pass(&all);
        writeln!(out)?;
        writeln!(out, "OVERALL: {}", if pass { "PASS" } else { "FAIL" })?;

        // Write a markdown summary file when an output directory is configured.
        if let Some(dir) = output_dir {
            if let Err(err) = write_summary_md(dir, &all, pass) {
                writeln!(out, "warning: could not write TestSummary.md: {err}")?;
            }
        }

        Ok(pass)
    }

    /// Ask one question, execute its retrieval actions, and gather the metrics.
    ///
    /// A question passes only when the model replied, at least one action
    /// parsed, and **every** retrieval it attempted succeeded. Any model error,
    /// parse failure, missing-but-needed client, or query error fails the
    /// question (with an explanatory note) but never aborts the other questions.
    ///
    /// The actual retrieval is delegated to [`PullDataController`] — the exact
    /// same code the live cloud app runs — so this self-test validates the real
    /// pull pass, not a copy of it. Returns the metrics **and** the raw
    /// artifacts so the caller can write them to disk for human review.
    fn probe_one(&self, question: &str) -> (QuestionMetrics, ProbeArtifacts) {
        let mut metrics = QuestionMetrics {
            question: question.to_string(),
            llm_ok: false,
            actions_parsed: 0,
            cosmos_actions: 0,
            cosmos_rows: 0,
            cosmos_bytes: 0,
            web_actions: 0,
            web_rows: 0,
            web_bytes: 0,
            container_details: Vec::new(),
            success: false,
            notes: Vec::new(),
        };
        let mut artifacts = ProbeArtifacts::empty();

        // --- LLM Call 1: ask the model what to retrieve -------------------
        let requested_at = now_rfc3339();
        let call1 = Call1Prompt::build(PROBE_USER_ID, question, "", &requested_at);
        let reply = match self.llm.complete(&call1.system, &call1.user) {
            // Self-tests only need the reply text, not the reported model.
            Ok(reply) => reply.content,
            Err(err) => {
                metrics.notes.push(format!("LLM Call 1 failed: {err}"));
                return (metrics, artifacts);
            }
        };
        metrics.llm_ok = true;
        artifacts.raw_reply = Some(reply.clone());

        // --- Retrieval via the shared pull controller --------------------
        // This is the live cloud path. We ask it to capture the per-query SQL
        // (`capture_queries = true`) so the `queries.json` artifact can prove
        // whether keyword or semantic search ran.
        let controller = PullDataController::new(
            self.cosmos.as_ref(),
            self.embedder.as_ref(),
            self.web.as_ref(),
        );
        let result = controller.execute(PROBE_USER_ID, question, &requested_at, &reply, true);

        // Save the assembled Response Data Context (what Call 2 would receive).
        artifacts.context = result.context;

        // The model returned no parseable actions.json: nothing to retrieve.
        let Some(plan) = result.plan else {
            metrics
                .notes
                .push("could not parse actions.json from the reply".to_string());
            metrics
                .notes
                .push(format!("raw LLM reply:\n{}", pretty_print_reply(&reply)));
            return (metrics, artifacts);
        };

        artifacts.actions = Some(plan.clone());
        metrics.actions_parsed = plan.actions.len();
        if plan.actions.is_empty() {
            // The model decided no data retrieval is needed (e.g. a simple
            // greeting). That is a valid outcome — not a failure.
            metrics
                .notes
                .push("no retrieval actions (none needed)".to_string());
            metrics.success = true;
            return (metrics, artifacts);
        }

        // --- Fold the controller's results into metrics + artifacts ------
        metrics.cosmos_actions = result.cosmos_actions;
        metrics.web_actions = result.web_actions;
        artifacts.queries = result.planned_queries;

        for group in result.groups {
            let rows = group.records.len();
            let bytes = values_bytes(&group.records);
            if group.container.eq_ignore_ascii_case("Web") {
                metrics.web_rows += rows;
                metrics.web_bytes += bytes;
            } else {
                metrics.cosmos_rows += rows;
                metrics.cosmos_bytes += bytes;
            }
            metrics.record_container(&group.container, rows, bytes);
            artifacts
                .results
                .push((group.action_id, group.container, group.records));
        }

        // Carry over the controller's per-action failure notes.
        metrics.notes.extend(result.notes);

        // A question is a success only when the model replied, something was
        // retrieved, and no attempted retrieval errored.
        metrics.success = metrics.llm_ok && metrics.actions_parsed > 0 && result.all_ok;
        (metrics, artifacts)
    }
}

/// Pretty-print the raw LLM reply for diagnostics.
///
/// If the reply contains valid JSON (array or object), format it with
/// indentation so a human can inspect the structure. Otherwise return the raw
/// text as-is.
fn pretty_print_reply(reply: &str) -> String {
    // Try to find and pretty-print the JSON portion.
    if let Some(start) = reply.find('[') {
        if let Some(end) = reply.rfind(']') {
            if let Some(json_str) = reply.get(start..=end) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    return serde_json::to_string_pretty(&value)
                        .unwrap_or_else(|_| reply.to_string());
                }
            }
        }
    }
    if let Some(start) = reply.find('{') {
        if let Some(end) = reply.rfind('}') {
            if let Some(json_str) = reply.get(start..=end) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    return serde_json::to_string_pretty(&value)
                        .unwrap_or_else(|_| reply.to_string());
                }
            }
        }
    }
    // Not valid JSON — return raw.
    reply.to_string()
}

/// Sum the serialized JSON byte size of a slice of retrieval records.
///
/// The records arrive from [`PullDataController`] already serialized to generic
/// JSON values (Cosmos documents and Brave results alike). Serialized length is
/// a faithful stand-in for "how much data came back": it counts the actual
/// payload (text + metadata) the model would consume.
fn values_bytes(values: &[serde_json::Value]) -> usize {
    values
        .iter()
        .map(|value| serde_json::to_string(value).map(|s| s.len()).unwrap_or(0))
        .sum()
}

/// Convert a byte count to kilobytes (1 KB = 1024 bytes).
fn bytes_to_kb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

/// Render the per-question metrics as a fixed-width summary table.
///
/// Columns: question number, LLM ok, actions parsed, Cosmos rows/KB, Brave
/// rows/KB, combined rows/KB, and the pass/fail result.
fn format_metrics_table(metrics: &[QuestionMetrics]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<3} {:<4} {:<7} {:<18} {:>6} {:>9} {:<6}",
        "#", "LLM", "Actions", "Container", "Rows", "KB", "Result",
    );
    let _ = writeln!(out, "{}", "-".repeat(60));
    for (index, m) in metrics.iter().enumerate() {
        let result_str = if m.success { "PASS" } else { "FAIL" };
        if m.container_details.is_empty() {
            let _ = writeln!(
                out,
                "{:<3} {:<4} {:<7} {:<18} {:>6} {:>9.2} {:<6}",
                index + 1,
                if m.llm_ok { "ok" } else { "ERR" },
                m.actions_parsed,
                "-",
                0,
                0.0_f64,
                result_str,
            );
        } else {
            for (ci, (container, rows, bytes)) in m.container_details.iter().enumerate() {
                if ci == 0 {
                    let _ = writeln!(
                        out,
                        "{:<3} {:<4} {:<7} {:<18} {:>6} {:>9.2} {:<6}",
                        index + 1,
                        if m.llm_ok { "ok" } else { "ERR" },
                        m.actions_parsed,
                        container,
                        rows,
                        bytes_to_kb(*bytes),
                        result_str,
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "{:<3} {:<4} {:<7} {:<18} {:>6} {:>9.2}",
                        "",
                        "",
                        "",
                        container,
                        rows,
                        bytes_to_kb(*bytes),
                    );
                }
            }
        }
    }
    out
}

/// The whole self-test passes only when every question passed.
fn overall_pass(metrics: &[QuestionMetrics]) -> bool {
    !metrics.is_empty() && metrics.iter().all(|m| m.success)
}

/// Write a `TestSummary.md` markdown file into `dir` with a per-question table.
fn write_summary_md(dir: &Path, metrics: &[QuestionMetrics], pass: bool) -> io::Result<()> {
    use std::fmt::Write as _;

    let mut md = String::new();
    let _ = writeln!(md, "# Data-Retrieval Self-Test Summary\n");
    let _ = writeln!(
        md,
        "**Overall: {}**\n",
        if pass { "PASS ✅" } else { "FAIL ❌" }
    );
    let _ = writeln!(
        md,
        "| # | Question | LLM | Actions | Containers | Rows | KB | Result |"
    );
    let _ = writeln!(
        md,
        "|---|----------|-----|---------|------------|------|----|--------|"
    );
    for (i, m) in metrics.iter().enumerate() {
        let containers: String = if m.container_details.is_empty() {
            "—".to_string()
        } else {
            m.container_details
                .iter()
                .map(|(name, rows, bytes)| {
                    format!("{name} ({rows} rows, {:.2} KB)", bytes_to_kb(*bytes))
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let result = if m.success { "PASS ✅" } else { "FAIL ❌" };
        // Truncate question for the table (first 60 chars).
        let q_display = if m.question.len() > 60 {
            format!("{}…", &m.question[..60])
        } else {
            m.question.clone()
        };
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {:.2} | {} |",
            i + 1,
            q_display,
            if m.llm_ok { "ok" } else { "ERR" },
            m.actions_parsed,
            containers,
            m.total_rows(),
            m.total_kb(),
            result,
        );
    }

    // Notes section for any failures.
    let has_notes = metrics.iter().any(|m| !m.notes.is_empty());
    if has_notes {
        let _ = writeln!(md, "\n## Notes\n");
        for (i, m) in metrics.iter().enumerate() {
            if !m.notes.is_empty() {
                let _ = writeln!(md, "**Q{}:**", i + 1);
                for note in &m.notes {
                    let _ = writeln!(md, "- {note}");
                }
                let _ = writeln!(md);
            }
        }
    }

    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("TestSummary.md"), md)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::SessionContext;

    /// Build a metrics row that is either a clean pass or a clean fail.
    fn metrics(success: bool) -> QuestionMetrics {
        QuestionMetrics {
            question: "q".to_string(),
            llm_ok: true,
            actions_parsed: 1,
            cosmos_actions: 1,
            cosmos_rows: 2,
            cosmos_bytes: 2048,
            web_actions: 1,
            web_rows: 3,
            web_bytes: 1024,
            container_details: vec![
                ("GaiaKB".to_string(), 2, 2048),
                ("Web".to_string(), 3, 1024),
            ],
            success,
            notes: Vec::new(),
        }
    }

    #[test]
    fn values_bytes_counts_serialized_payloads() {
        // A populated record serializes to a non-trivial number of bytes.
        let record = serde_json::json!({
            "id": "GaiaKB|rust|2026-05-10",
            "data": "the borrow checker is strict",
        });
        assert!(values_bytes(std::slice::from_ref(&record)) > 0);
        // An empty slice has zero size.
        assert_eq!(values_bytes(&[]), 0);
    }

    #[test]
    fn bytes_to_kb_divides_by_1024() {
        assert_eq!(bytes_to_kb(0), 0.0);
        assert_eq!(bytes_to_kb(1024), 1.0);
        assert_eq!(bytes_to_kb(512), 0.5);
    }

    #[test]
    fn total_rows_and_kb_combine_cosmos_and_web() {
        let m = metrics(true);
        assert_eq!(m.total_rows(), 5);
        // 2048 + 1024 bytes = 3 KB.
        assert_eq!(m.total_kb(), 3.0);
    }

    #[test]
    fn overall_pass_requires_every_question_to_pass() {
        // All passing -> pass.
        assert!(overall_pass(&[metrics(true), metrics(true)]));
        // Any failure -> fail.
        assert!(!overall_pass(&[metrics(true), metrics(false)]));
        // No questions at all is not a pass (nothing was validated).
        assert!(!overall_pass(&[]));
    }

    #[test]
    fn format_metrics_table_has_a_header_and_a_row_per_question() {
        let table = format_metrics_table(&[metrics(true), metrics(false)]);
        // Header columns are present.
        assert!(table.contains("Container"));
        assert!(table.contains("Rows"));
        assert!(table.contains("Result"));
        // Container names appear in the table.
        assert!(table.contains("GaiaKB"));
        assert!(table.contains("Web"));
        // One PASS row and one FAIL row.
        assert!(table.contains("PASS"));
        assert!(table.contains("FAIL"));
    }

    #[test]
    fn write_to_clears_stale_json_files_before_writing_new_artifacts() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "gaia_data_retrieval_probe_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&dir).expect("temp artifact directory should be created");

        std::fs::write(dir.join("stale.json"), "{\"stale\":true}")
            .expect("stale json should be written");
        std::fs::write(dir.join("keep.txt"), "keep me")
            .expect("non-json marker file should be written");

        let artifacts = ProbeArtifacts {
            raw_reply: Some("{\"message\":\"hello\"}".to_string()),
            actions: Some(ActionsFile {
                version: "1.0".to_string(),
                session: SessionContext {
                    user_id: PROBE_USER_ID.to_string(),
                    requested_at: "2026-06-20T00:00:00Z".to_string(),
                },
                actions: Vec::new(),
            }),
            queries: Vec::new(),
            results: Vec::new(),
            context: "# Response Data Context\n\n## WebSearchResults\n".to_string(),
        };

        artifacts
            .write_to(&dir)
            .expect("probe artifacts should write successfully");

        assert!(dir.join("reply.json").exists());
        assert!(dir.join("actions.json").exists());
        assert!(dir.join("responsedatacontext.md").exists());
        assert!(!dir.join("stale.json").exists());
        assert!(dir.join("keep.txt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pretty_print_reply_handles_arrays_objects_and_raw_text() {
        // An embedded JSON array is pretty-printed.
        let array = pretty_print_reply("noise [ {\"id\":\"q1\"} ] tail");
        assert!(array.contains("\"id\": \"q1\""));
        // An embedded JSON object (no array) takes the object branch.
        let object = pretty_print_reply("prefix {\"message\":\"hi\"} suffix");
        assert!(object.contains("\"message\": \"hi\""));
        // Text with no JSON is returned unchanged.
        assert_eq!(pretty_print_reply("just words"), "just words");
    }

    #[test]
    fn write_summary_md_renders_table_and_notes_section() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "gaia_retrieval_summary_{}_{}",
            std::process::id(),
            unique
        ));

        // A failing question with a note exercises the notes section.
        let mut failing = metrics(false);
        failing.notes = vec!["Cosmos action q1 failed".to_string()];
        write_summary_md(&dir, &[metrics(true), failing], false).expect("write summary");

        let md = std::fs::read_to_string(dir.join("TestSummary.md")).expect("read summary");
        assert!(md.contains("# Data-Retrieval Self-Test Summary"));
        assert!(md.contains("FAIL ❌"));
        // The per-question table and the notes section are both present.
        assert!(md.contains("GaiaKB (2 rows"));
        assert!(md.contains("## Notes"));
        assert!(md.contains("Cosmos action q1 failed"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
