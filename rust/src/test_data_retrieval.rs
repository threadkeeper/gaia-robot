//! The [`DataRetrievalProbe`] type: an end-to-end self-test of Gaia's pull pass.
//!
//! This module powers the `gaia-robot test-data-retrieval` subcommand (and the
//! `infra/TestDataRetrieval.ps1` wrapper). Its job is to prove that the whole
//! *retrieval* half of a turn actually works against live infrastructure before
//! we ship a new build:
//!
//! 1. Ask the model **five** fixed questions of varying length and subject, each
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

use crate::actions::{ActionPlan, ActionsFile, SessionContext};
use crate::cosmos::CosmosClient;
use crate::executor::Executor;
use crate::llm::LlmClient;
use crate::prompt::{now_rfc3339, Call1Prompt};
use crate::search_history::SearchResult;
use crate::storage::Record;
use crate::web_search::BraveClient;

/// The user every probe question is scoped to. Matches the `threadkeeper`
/// exports under `migrations/`, so the queries run against real seeded data.
const PROBE_USER_ID: &str = "threadkeeper";

/// The five fixed probe questions, chosen to vary in **length** and **subject**
/// so the model authors a spread of retrieval actions (personal recall, durable
/// facts, fresh web facts, and relationship/diary lookups).
const PROBE_QUESTIONS: [&str; 5] = [
    // 1. Very short, everyday.
    "How are you today?",
    // 2. Medium, personal recall (UsersDataLake / GaiaDataLake territory).
    "Remind me what we talked about regarding the robot's adventures in the forest recently.",
    // 3. Long, multi-topic personal synthesis (facts + history).
    "Can you summarise everything you know about my interests in music, books, and the \
     outdoors, note how any of those have shifted over the past month, and tie that back to \
     anything specific from our recent conversations?",
    // 4. Factual, fresh — should trigger a Web search.
    "What are the latest developments in Mars exploration this year?",
    // 5. Relationship / diary lookup.
    "How has our friendship been going lately, and is there anything you noted in your diary \
     about me?",
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
    /// Per-action retrieval results keyed by `(action_id, container)`.
    results: Vec<(String, String, Vec<serde_json::Value>)>,
}

impl ProbeArtifacts {
    fn empty() -> Self {
        Self {
            raw_reply: None,
            actions: None,
            results: Vec::new(),
        }
    }

    /// Write the artifacts as pretty-printed JSON files into `dir`.
    ///
    /// Creates the directory if it does not exist. Overwrites any existing files.
    fn write_to(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;

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

        for (action_id, container, records) in &self.results {
            let filename = format!("{action_id}_{container}.json");
            let path = dir.join(filename);
            let json = serde_json::to_string_pretty(records).unwrap_or_default();
            std::fs::write(&path, json)?;
        }

        Ok(())
    }
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
        let web = BraveClient::from_env();

        Ok(Self { llm, cosmos, web })
    }

    /// Run probe questions, write a report to `out`, and return whether the
    /// self-test passed.
    ///
    /// When `only` is `Some(n)`, run only question `n` (1-based). When `None`,
    /// run all five. When `output_dir` is set, write pretty-printed JSON
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
                let q_dir = dir.join(format!("q{}", index + 1));
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
        Ok(pass)
    }

    /// Ask one question, execute its retrieval actions, and gather the metrics.
    ///
    /// A question passes only when the model replied, at least one action
    /// parsed, and **every** retrieval it attempted succeeded. Any model error,
    /// parse failure, missing-but-needed client, or query error fails the
    /// question (with an explanatory note) but never aborts the other questions.
    ///
    /// Returns the metrics **and** the raw artifacts so the caller can write
    /// them to disk for human review.
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
            Ok(reply) => reply,
            Err(err) => {
                metrics.notes.push(format!("LLM Call 1 failed: {err}"));
                return (metrics, artifacts);
            }
        };
        metrics.llm_ok = true;
        artifacts.raw_reply = Some(reply.clone());

        // --- Parse actions.json out of the reply --------------------------
        let actions = match crate::actions::parse_call1_actions(&reply) {
            Some(actions) => actions,
            None => {
                metrics
                    .notes
                    .push("could not parse actions.json from the reply".to_string());
                metrics
                    .notes
                    .push(format!("raw LLM reply:\n{}", pretty_print_reply(&reply)));
                return (metrics, artifacts);
            }
        };
        artifacts.actions = Some(actions.clone());
        metrics.actions_parsed = actions.actions.len();
        if actions.actions.is_empty() {
            // The model decided no data retrieval is needed (e.g. a simple
            // greeting). That is a valid outcome — not a failure.
            metrics
                .notes
                .push("no retrieval actions (none needed)".to_string());
            metrics.success = true;
            return (metrics, artifacts);
        }

        // Split the plan into the Cosmos-backed queries and the Web searches.
        let (web_actions, cosmos_actions): (Vec<ActionPlan>, Vec<ActionPlan>) = actions
            .actions
            .into_iter()
            .partition(|action| action.target.eq_ignore_ascii_case("Web"));

        // Track whether every attempted retrieval succeeded.
        let mut all_ok = true;

        // --- Cosmos retrieval --------------------------------------------
        metrics.cosmos_actions = cosmos_actions.len();
        if !cosmos_actions.is_empty() {
            match &self.cosmos {
                Some(client) => {
                    // Save target names before moving actions into the plan.
                    let targets: Vec<String> =
                        cosmos_actions.iter().map(|a| a.target.clone()).collect();
                    let action_ids: Vec<String> =
                        cosmos_actions.iter().map(|a| a.id.clone()).collect();
                    let plan = ActionsFile {
                        version: actions.version.clone(),
                        session: SessionContext {
                            user_id: PROBE_USER_ID.to_string(),
                            requested_at: requested_at.clone(),
                        },
                        actions: cosmos_actions,
                    };
                    let outcomes = Executor::new(client).run(&plan);
                    for ((target, action_id), outcome) in
                        targets.iter().zip(action_ids.iter()).zip(outcomes.iter())
                    {
                        match &outcome.result {
                            Ok(records) => {
                                let rows = records.len();
                                let bytes = records_bytes(records);
                                metrics.cosmos_rows += rows;
                                metrics.cosmos_bytes += bytes;
                                metrics.record_container(target, rows, bytes);
                                // Collect result artifacts as generic JSON values.
                                let values: Vec<serde_json::Value> = records
                                    .iter()
                                    .filter_map(|r| serde_json::to_value(r).ok())
                                    .collect();
                                artifacts
                                    .results
                                    .push((action_id.clone(), target.clone(), values));
                            }
                            Err(err) => {
                                all_ok = false;
                                metrics
                                    .notes
                                    .push(format!("Cosmos action {} failed: {err}", outcome.id));
                            }
                        }
                    }
                }
                None => {
                    all_ok = false;
                    metrics.notes.push(format!(
                        "{} Cosmos action(s) requested but Cosmos is not configured",
                        metrics.cosmos_actions
                    ));
                }
            }
        }

        // --- Web (Brave) retrieval ---------------------------------------
        metrics.web_actions = web_actions.len();
        if !web_actions.is_empty() {
            match &self.web {
                Some(client) => {
                    for action in &web_actions {
                        let query = web_query_for(action, question);
                        match client.search(&query, action.effective_top()) {
                            Ok(results) => {
                                let rows = results.len();
                                let bytes = results_bytes(&results);
                                metrics.web_rows += rows;
                                metrics.web_bytes += bytes;
                                metrics.record_container("Web", rows, bytes);
                                // Collect Brave results as generic JSON values.
                                let values: Vec<serde_json::Value> = results
                                    .iter()
                                    .filter_map(|r| serde_json::to_value(r).ok())
                                    .collect();
                                artifacts.results.push((
                                    action.id.clone(),
                                    "Web".to_string(),
                                    values,
                                ));
                            }
                            Err(err) => {
                                all_ok = false;
                                metrics
                                    .notes
                                    .push(format!("Brave action {} failed: {err}", action.id));
                            }
                        }
                    }
                }
                None => {
                    all_ok = false;
                    metrics.notes.push(format!(
                        "{} Web action(s) requested but Brave is not configured",
                        metrics.web_actions
                    ));
                }
            }
        }

        // A question is a success only when the model replied, something was
        // retrieved, and no attempted retrieval errored.
        metrics.success = metrics.llm_ok && metrics.actions_parsed > 0 && all_ok;
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

/// Choose the text to search the web with for a `Web` action.
///
/// The model's free-text `text` filter is the most precise signal, then its
/// `intent`; if neither is present we fall back to the user's question so a Web
/// action always has *something* to search for.
fn web_query_for(action: &ActionPlan, question: &str) -> String {
    if let Some(text) = action
        .filters
        .text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        return text.to_string();
    }
    let intent = action.intent.trim();
    if !intent.is_empty() {
        return intent.to_string();
    }
    question.trim().to_string()
}

/// Sum the serialized JSON byte size of a slice of Cosmos records.
///
/// Serialized length is a faithful stand-in for "how much data came back": it
/// counts the actual document payload (text + metadata) the model would consume.
fn records_bytes(records: &[Record]) -> usize {
    records
        .iter()
        .map(|record| serde_json::to_string(record).map(|s| s.len()).unwrap_or(0))
        .sum()
}

/// Sum the serialized JSON byte size of a slice of Brave search results.
fn results_bytes(results: &[SearchResult]) -> usize {
    results
        .iter()
        .map(|result| serde_json::to_string(result).map(|s| s.len()).unwrap_or(0))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ActionFilters;

    /// Build a minimal action for the pure-helper tests.
    fn action(target: &str, intent: &str, text: Option<&str>) -> ActionPlan {
        ActionPlan {
            id: "q1".to_string(),
            kind: "query".to_string(),
            target: target.to_string(),
            user_id: None,
            entity: Some(PROBE_USER_ID.to_string()),
            intent: intent.to_string(),
            top: 3,
            query: None,
            filters: ActionFilters {
                text: text.map(str::to_string),
                ..ActionFilters::default()
            },
        }
    }

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
    fn web_query_prefers_text_then_intent_then_question() {
        // The free-text filter wins when present.
        let with_text = action("Web", "broad intent", Some("  mars rovers  "));
        assert_eq!(web_query_for(&with_text, "the question"), "mars rovers");

        // Otherwise the intent is used.
        let with_intent = action("Web", "latest mars news", None);
        assert_eq!(
            web_query_for(&with_intent, "the question"),
            "latest mars news"
        );

        // With neither, fall back to the user's question.
        let bare = action("Web", "   ", None);
        assert_eq!(web_query_for(&bare, "  the question  "), "the question");
    }

    #[test]
    fn record_and_result_bytes_count_serialized_payloads() {
        let record = Record::new(
            "GaiaKB|rust|2026-05-10",
            "rust",
            "",
            "2026-05-10",
            crate::storage::RecordKind::KnowledgeBase,
            "the borrow checker is strict",
            Vec::new(),
        );
        // A populated record serializes to a non-trivial number of bytes.
        assert!(records_bytes(std::slice::from_ref(&record)) > 0);
        // An empty slice has zero size.
        assert_eq!(records_bytes(&[]), 0);

        let result = SearchResult::new("Title", "https://example.com", "snippet");
        assert!(results_bytes(std::slice::from_ref(&result)) > 0);
        assert_eq!(results_bytes(&[]), 0);
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
}
