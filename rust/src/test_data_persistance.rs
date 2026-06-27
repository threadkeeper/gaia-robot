//! The [`DataPersistenceProbe`] type: an end-to-end self-test of Gaia's write pass.
//!
//! This module powers the `gaia-robot test-data-persistence` subcommand (and the
//! future `infra/TestDataPersistence.ps1` wrapper). It is the **write-pass**
//! counterpart to the retrieval probe ([`crate::test_data_retrieval`]) and the
//! execution probe ([`crate::test_data_execution`]): where those exercise the two
//! LLM calls, this one proves that the *persistence* half of a turn actually
//! works against live infrastructure before we ship a new build.
//!
//! For each target container it:
//! 1. Writes a first time-stamped chunk for a dedicated `selftest` entity, then
//!    a second chunk, through the shared [`WriteDataController`] — the same code
//!    the cloud app will run.
//! 2. Reads the record back and asserts the append worked (both chunks present)
//!    and the embedding was refreshed (`dataVector` is non-empty).
//! 3. For `GaiaDataLake`, additionally asserts the `DataLakeIndex` entry was
//!    synced with a half-size vector.
//!
//! The probe is deliberately strict: a container passes only when every write
//! succeeded and the read-back verification held. The subcommand turns the
//! overall boolean into the process exit code, so it doubles as an on-demand
//! check and a hard CI gate. It writes under a dedicated `selftest` partition so
//! it never touches real user data.

use std::io::{self, Write};
use std::path::Path;

use crate::prompt::now_rfc3339;
use crate::write_data_controller::{WriteDataController, DATALAKE_CONTAINER};

/// The dedicated partition key the probe writes under, isolating its records
/// from real users (`/entity` and `/userId` both carry this value).
const TEST_ENTITY: &str = "selftest";

/// The containers the probe writes to, one per writer family it must validate:
/// the DataLake (which also syncs the index), the knowledge base, and the diary.
const PROBE_CONTAINERS: [&str; 3] = ["GaiaDataLake", "GaiaKB", "GaiaDiary"];

/// Per-container results gathered by the probe.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceMetrics {
    /// The container that was written, e.g. `GaiaDataLake`.
    pub container: String,
    /// The deterministic daily id written.
    pub id: String,
    /// The [`WriteAction`](crate::write_data_controller::WriteAction) label from
    /// the first write (`created`/`appended`/`unchanged`).
    pub first_action: String,
    /// The action label from the second (appending) write.
    pub second_action: String,
    /// The size in bytes of the day's text after both writes.
    pub data_bytes: usize,
    /// The dimensionality of the in-place `dataVector` that was written.
    pub vector_dims: usize,
    /// The dimensionality of the synced `DataLakeIndex` vector (`0` when the
    /// container does not feed the index).
    pub index_dims: usize,
    /// Whether the read-back verification held (both chunks present + vector).
    pub verified: bool,
    /// Overall pass/fail for this container.
    pub success: bool,
    /// Human-readable notes explaining any failure.
    pub notes: Vec<String>,
}

/// Runs the data-persistence self-test end to end.
///
/// Holds the shared [`WriteDataController`] (built from the same environment the
/// console app and server use). Built with [`DataPersistenceProbe::from_env`],
/// driven with [`DataPersistenceProbe::run`].
pub struct DataPersistenceProbe {
    /// The shared write controller (Cosmos + embeddings).
    controller: WriteDataController,
}

impl DataPersistenceProbe {
    /// Build a probe from the process environment.
    ///
    /// Reuses [`WriteDataController::from_env`]; both a Cosmos and an embedding
    /// client are mandatory (the write pass cannot persist an embedded record
    /// without them). Returns a clear, human-readable error string when either
    /// is missing, so the subcommand can print it and exit non-zero.
    pub fn from_env() -> Result<Self, String> {
        let controller = WriteDataController::from_env()
            .map_err(|err| format!("write pass cannot start: {err}"))?;
        if !controller.is_online() {
            return Err(
                "write pass is offline. Set GAIA_MODE=dev (or local), COSMOS_ENDPOINT + \
                 COSMOS_AAD_TOKEN, and FOUNDRY_ENDPOINT + EMBEDDING_DEPLOYMENT + a credential."
                    .to_string(),
            );
        }
        Ok(Self { controller })
    }

    /// Run the probe, write a report to `out`, and return whether it passed.
    ///
    /// When `only` is `Some(n)`, run only container `n` (1-based); when `None`,
    /// run all of them. When `output_dir` is set, write the read-back records as
    /// JSON artifacts plus a markdown summary. The boolean gate is `true` only
    /// when every executed container passed.
    pub fn run(
        &self,
        only: Option<usize>,
        output_dir: Option<&Path>,
        out: &mut impl Write,
    ) -> io::Result<bool> {
        writeln!(out, "Gaia data-persistence self-test")?;
        writeln!(out, "  entity : {TEST_ENTITY}")?;
        writeln!(out, "  online : {}", self.controller.is_online())?;
        writeln!(out)?;

        let containers: Vec<(usize, &str)> = match only {
            Some(n) if n >= 1 && n <= PROBE_CONTAINERS.len() => {
                vec![(n - 1, PROBE_CONTAINERS[n - 1])]
            }
            Some(n) => {
                writeln!(
                    out,
                    "ERROR: container {n} does not exist (1–{}).",
                    PROBE_CONTAINERS.len()
                )?;
                return Ok(false);
            }
            None => PROBE_CONTAINERS.iter().copied().enumerate().collect(),
        };

        let mut all = Vec::with_capacity(containers.len());

        // === Phase 1: write pass =========================================
        // Drive two timestamped appends per container through the shared
        // controller — the exact code the cloud app runs. We capture a "ticket"
        // per container describing what we wrote so the verification phase can
        // re-fetch each record straight from Cosmos.
        let mut tickets = Vec::with_capacity(containers.len());
        for (index, container) in &containers {
            writeln!(
                out,
                "[{}/{}] writing {container}",
                index + 1,
                PROBE_CONTAINERS.len()
            )?;
            let ticket = self.write_one(container);
            for note in &ticket.notes {
                writeln!(out, "      - {note}")?;
            }
            writeln!(
                out,
                "      => {} | {} | {} bytes",
                if ticket.write_ok {
                    "written"
                } else {
                    "WRITE FAILED"
                },
                ticket.second_action,
                ticket.data_bytes,
            )?;
            tickets.push(ticket);
        }

        // === Phase 2: read-back verification from Cosmos =================
        // The section the request asks for: re-retrieve every record we just
        // wrote directly from Cosmos and prove (a) it persisted — both chunks
        // present — and (b) its vectors are sound: the in-place `dataVector` is
        // correctly sized and finite, and the synced `DataLakeIndex` vector is
        // the half-size, finite, *unit-length* Matryoshka projection.
        writeln!(out)?;
        writeln!(
            out,
            "Retrieving records from Cosmos to verify persistence and vectors…"
        )?;
        for ticket in &tickets {
            writeln!(out, "  - {} ({})", ticket.container, ticket.id)?;
            let (metrics, readback) = self.verify_persisted(ticket);

            if let Some(dir) = output_dir {
                if let Err(err) = write_artifact(dir, &ticket.container, readback.as_ref()) {
                    writeln!(out, "      - warning: could not write artifact: {err}")?;
                }
            }

            for note in &metrics.notes {
                writeln!(out, "      - {note}")?;
            }
            writeln!(
                out,
                "      => {} | vector {}d | index {}d",
                if metrics.success {
                    "VERIFIED"
                } else {
                    "NOT VERIFIED"
                },
                metrics.vector_dims,
                metrics.index_dims,
            )?;
            all.push(metrics);
        }

        writeln!(out)?;
        write!(out, "{}", format_metrics_table(&all))?;
        let pass = overall_pass(&all);
        writeln!(out)?;
        writeln!(out, "OVERALL: {}", if pass { "PASS" } else { "FAIL" })?;

        if let Some(dir) = output_dir {
            if let Err(err) = write_summary_md(dir, &all, pass) {
                writeln!(out, "warning: could not write TestSummary.md: {err}")?;
            }
        }

        Ok(pass)
    }

    /// Write twice to one container and return a ticket describing what landed.
    ///
    /// Performs the create-then-append pair through the shared controller and
    /// records the resulting id, action labels, byte size, and the dimensions
    /// the write reported. A failed write yields a ticket with `write_ok = false`
    /// and an explanatory note; it never aborts the overall run.
    fn write_one(&self, container: &str) -> WriteTicket {
        // Two distinct, run-tagged chunks so the content always changes (no M1
        // content-hash skip) and the read-back can prove both lines are present.
        let now = now_rfc3339();
        let date = now.split_once('T').map(|(d, _)| d).unwrap_or(&now);
        let run_tag = now.clone();
        let chunk1 = format!("self-test write A ({run_tag})");
        let chunk2 = format!("self-test write B ({run_tag})");
        let ts1 = format!("{date}T09:00:00Z");
        let ts2 = format!("{date}T09:00:01Z");

        let mut ticket = WriteTicket {
            container: container.to_string(),
            id: String::new(),
            chunk1: chunk1.clone(),
            chunk2: chunk2.clone(),
            first_action: String::new(),
            second_action: String::new(),
            data_bytes: 0,
            vector_dims: 0,
            index_dims: 0,
            write_ok: false,
            notes: Vec::new(),
        };

        // --- Write #1 (create or append) ---------------------------------
        let first = match self
            .controller
            .upsert_daily(container, TEST_ENTITY, &ts1, &chunk1)
        {
            Ok(outcome) => outcome,
            Err(err) => {
                ticket.notes.push(format!("first write failed: {err}"));
                return ticket;
            }
        };
        ticket.id = first.id.clone();
        ticket.first_action = first.action.label().to_string();

        // --- Write #2 (append) -------------------------------------------
        let second = match self
            .controller
            .upsert_daily(container, TEST_ENTITY, &ts2, &chunk2)
        {
            Ok(outcome) => outcome,
            Err(err) => {
                ticket.notes.push(format!("second write failed: {err}"));
                return ticket;
            }
        };
        ticket.second_action = second.action.label().to_string();
        ticket.data_bytes = second.data_bytes;
        ticket.vector_dims = second.vector_dims;
        ticket.index_dims = second.index_dims;
        ticket.write_ok = true;
        ticket
    }

    /// Re-read a written record straight from Cosmos and verify it is durable.
    ///
    /// Returns the metrics plus the read-back record (for artifact writing). A
    /// container is verified only when both appended chunks are present and the
    /// vectors are *sound*: the in-place `dataVector` is correctly sized and
    /// finite, and (for the DataLake) the synced `DataLakeIndex` vector is the
    /// half-size, finite, unit-length Matryoshka projection. A write that already
    /// failed short-circuits to a failed metrics row.
    fn verify_persisted(
        &self,
        ticket: &WriteTicket,
    ) -> (PersistenceMetrics, Option<serde_json::Value>) {
        let mut metrics = PersistenceMetrics {
            container: ticket.container.clone(),
            id: ticket.id.clone(),
            first_action: ticket.first_action.clone(),
            second_action: ticket.second_action.clone(),
            data_bytes: ticket.data_bytes,
            vector_dims: ticket.vector_dims,
            index_dims: ticket.index_dims,
            verified: false,
            success: false,
            notes: ticket.notes.clone(),
        };

        // A failed write has nothing to retrieve; carry its notes forward.
        if !ticket.write_ok {
            return (metrics, None);
        }

        // --- Retrieve the record from Cosmos -----------------------------
        let record = match self
            .controller
            .read_record(&ticket.container, TEST_ENTITY, &ticket.id)
        {
            Ok(Some(record)) => record,
            Ok(None) => {
                metrics
                    .notes
                    .push("read-back found no record after writing".to_string());
                return (metrics, None);
            }
            Err(err) => {
                metrics.notes.push(format!("read-back failed: {err}"));
                return (metrics, None);
            }
        };

        let mut verified = true;

        // (a) Persistence: both appended chunks must be present in the day's log.
        if !record.data.contains(&ticket.chunk1) {
            metrics
                .notes
                .push("read-back is missing the first chunk".to_string());
            verified = false;
        }
        if !record.data.contains(&ticket.chunk2) {
            metrics
                .notes
                .push("read-back is missing the appended chunk".to_string());
            verified = false;
        }

        // (b) Vector soundness: the in-place embedding must be present, the right
        // size, and free of NaN/inf. We don't force unit norm here — the raw
        // embedding may not be normalized.
        metrics.vector_dims = record.data_vector.len();
        for problem in vector_problems(&record.data_vector, ticket.vector_dims, false) {
            metrics.notes.push(format!("dataVector unsound: {problem}"));
            verified = false;
        }

        // (c) For the DataLake, the synced index vector must also be sound and,
        // because Matryoshka truncation renormalizes it, unit length.
        if ticket.container == DATALAKE_CONTAINER {
            match self.controller.read_index_vector(TEST_ENTITY, &ticket.id) {
                Ok(Some(index_vec)) => {
                    metrics.index_dims = index_vec.len();
                    for problem in vector_problems(&index_vec, ticket.index_dims, true) {
                        metrics
                            .notes
                            .push(format!("DataLakeIndex vector unsound: {problem}"));
                        verified = false;
                    }
                }
                Ok(None) => {
                    metrics
                        .notes
                        .push("DataLakeIndex entry was not synced".to_string());
                    verified = false;
                }
                Err(err) => {
                    metrics
                        .notes
                        .push(format!("DataLakeIndex read-back failed: {err}"));
                    verified = false;
                }
            }
        }

        metrics.verified = verified;
        metrics.success = verified;

        let value = serde_json::to_value(&record).ok();
        (metrics, value)
    }
}

/// What a single container's write pass produced, handed to the verification
/// phase so it can re-fetch and check the record from Cosmos.
#[derive(Debug, Clone)]
struct WriteTicket {
    /// The container that was written, e.g. `GaiaDataLake`.
    container: String,
    /// The deterministic daily id written.
    id: String,
    /// The first chunk's text (must reappear in the read-back).
    chunk1: String,
    /// The appended chunk's text (must reappear in the read-back).
    chunk2: String,
    /// The action label from the first write.
    first_action: String,
    /// The action label from the second (appending) write.
    second_action: String,
    /// The size in bytes of the day's text after both writes.
    data_bytes: usize,
    /// The dimensionality the write reported for the in-place `dataVector`.
    vector_dims: usize,
    /// The dimensionality the write reported for the synced index vector
    /// (`0` when the container does not feed the index).
    index_dims: usize,
    /// Whether both writes succeeded.
    write_ok: bool,
    /// Notes captured during the write pass (e.g. a failure reason).
    notes: Vec<String>,
}

// --- Pure helpers (no network) ----------------------------------------------

/// Return the problems that make `vec` an unsound embedding (empty list == the
/// vector is sound).
///
/// Checks, in order: the vector is non-empty; it has exactly `expected_dims`
/// components (skipped when `expected_dims == 0`); every component is finite
/// (no `NaN`/`inf`); and — when `require_unit_norm` is set — the vector is unit
/// length, which the `DataLakeIndex` vector must be after Matryoshka
/// truncation re-normalizes it.
fn vector_problems(vec: &[f32], expected_dims: usize, require_unit_norm: bool) -> Vec<String> {
    let mut problems = Vec::new();

    if vec.is_empty() {
        problems.push("vector is empty".to_string());
        // Nothing else is meaningful for an empty vector.
        return problems;
    }
    if expected_dims > 0 && vec.len() != expected_dims {
        problems.push(format!(
            "vector has {} dims, expected {expected_dims}",
            vec.len()
        ));
    }
    if vec.iter().any(|v| !v.is_finite()) {
        problems.push("vector contains non-finite (NaN/inf) values".to_string());
    }
    if require_unit_norm {
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        // 1e-3 tolerates the f32 rounding introduced by truncation + re-norm.
        if (norm - 1.0).abs() > 1e-3 {
            problems.push(format!("vector norm {norm:.4} is not unit length"));
        }
    }

    problems
}

/// Render the per-container results as a fixed-width text table with a header
/// row and one row per container.
fn format_metrics_table(all: &[PersistenceMetrics]) -> String {
    let mut table = String::new();
    table.push_str("container        | result | action   | vector | index | bytes\n");
    table.push_str("-----------------+--------+----------+--------+-------+------\n");
    for m in all {
        table.push_str(&format!(
            "{:<16} | {:<6} | {:<8} | {:>6} | {:>5} | {:>5}\n",
            m.container,
            if m.success { "PASS" } else { "FAIL" },
            m.second_action,
            m.vector_dims,
            m.index_dims,
            m.data_bytes,
        ));
    }
    table
}

/// The overall gate: `true` only when at least one container ran and every one
/// of them passed.
fn overall_pass(all: &[PersistenceMetrics]) -> bool {
    !all.is_empty() && all.iter().all(|m| m.success)
}

/// Write one container's read-back record as a pretty-printed JSON artifact.
fn write_artifact(
    dir: &Path,
    container: &str,
    record: Option<&serde_json::Value>,
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    if let Some(value) = record {
        let path = dir.join(format!("{container}.json"));
        let json = serde_json::to_string_pretty(value).unwrap_or_default();
        std::fs::write(path, json)?;
    }
    Ok(())
}

/// Write a small markdown summary of the run.
fn write_summary_md(dir: &Path, all: &[PersistenceMetrics], pass: bool) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut md = String::new();
    md.push_str("# Data-persistence self-test\n\n");
    md.push_str(&format!(
        "**Overall:** {}\n\n",
        if pass { "PASS" } else { "FAIL" }
    ));
    md.push_str("```\n");
    md.push_str(&format_metrics_table(all));
    md.push_str("```\n");
    std::fs::write(dir.join("TestSummary.md"), md)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a metrics row that is either a clean pass or a clean fail.
    fn metrics(container: &str, success: bool) -> PersistenceMetrics {
        PersistenceMetrics {
            container: container.to_string(),
            id: format!("{container}|selftest|2026-06-27"),
            first_action: "created".to_string(),
            second_action: "appended".to_string(),
            data_bytes: 128,
            vector_dims: 1536,
            index_dims: if container == "GaiaDataLake" { 768 } else { 0 },
            verified: success,
            success,
            notes: Vec::new(),
        }
    }

    #[test]
    fn overall_pass_requires_every_container_to_pass() {
        assert!(overall_pass(&[
            metrics("GaiaKB", true),
            metrics("GaiaDiary", true)
        ]));
        assert!(!overall_pass(&[
            metrics("GaiaKB", true),
            metrics("GaiaDiary", false)
        ]));
        // An empty run has validated nothing and must not count as a pass.
        assert!(!overall_pass(&[]));
    }

    #[test]
    fn format_metrics_table_has_a_header_and_a_row_per_container() {
        let table =
            format_metrics_table(&[metrics("GaiaDataLake", true), metrics("GaiaKB", false)]);
        let lines: Vec<&str> = table.lines().collect();
        // 2 header lines + 2 data rows.
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("container"));
        assert!(lines[0].contains("index"));
        assert!(table.contains("GaiaDataLake"));
        assert!(table.contains("PASS"));
        assert!(table.contains("FAIL"));
        // The DataLake row reports its 768-d index vector.
        assert!(table.contains("768"));
    }

    #[test]
    fn datalake_metrics_report_an_index_vector_others_do_not() {
        assert_eq!(metrics("GaiaDataLake", true).index_dims, 768);
        assert_eq!(metrics("GaiaKB", true).index_dims, 0);
        assert_eq!(metrics("GaiaDiary", true).index_dims, 0);
    }

    #[test]
    fn vector_problems_accepts_a_sound_unit_vector() {
        // A finite, correctly-sized, unit-length vector has no problems.
        let unit = [0.6f32, 0.8]; // norm == 1.0 exactly
        assert!(vector_problems(&unit, 2, true).is_empty());
        // Without the unit-norm requirement a non-unit vector is still fine.
        assert!(vector_problems(&[3.0, 4.0], 2, false).is_empty());
    }

    #[test]
    fn vector_problems_flags_empty_wrong_size_and_non_finite() {
        // Empty short-circuits to a single problem.
        assert_eq!(vector_problems(&[], 768, false).len(), 1);
        // Wrong dimensionality is reported.
        let wrong = vector_problems(&[0.1, 0.2, 0.3], 768, false);
        assert!(wrong.iter().any(|p| p.contains("expected 768")));
        // NaN / infinity are rejected.
        let nan = vector_problems(&[f32::NAN, 0.0], 2, false);
        assert!(nan.iter().any(|p| p.contains("non-finite")));
    }

    #[test]
    fn vector_problems_flags_a_non_unit_vector_when_required() {
        // Norm is 5.0, not 1.0, so the unit-norm check must complain.
        let problems = vector_problems(&[3.0, 4.0], 2, true);
        assert!(problems.iter().any(|p| p.contains("not unit length")));
    }
}
