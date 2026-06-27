//! The [`HealthReport`] type: a structured readiness report for the server's
//! `/readyz` endpoint.
//!
//! `/healthz` stays a cheap liveness probe (it only proves the process is up and
//! accepting connections). `/readyz` is the *deep* check: it actively probes
//! every external dependency the running engine is wired to — Cosmos DB, the
//! Foundry model-router, the Foundry embedding deployment, and Brave Search —
//! exercising their real connectivity and RBAC. The result is collected here so
//! the server can serialise it to JSON and choose an overall HTTP status.

use serde::Serialize;

/// The outcome of probing a single dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    /// The dependency was probed and responded successfully.
    Ok,
    /// The dependency is configured but the probe failed (see `detail`).
    Failed,
    /// The dependency is not configured in this deployment, so it was not probed.
    Skipped,
}

/// The health of one external dependency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DependencyHealth {
    /// Stable identifier, e.g. `"cosmos"` or `"foundry-model-router"`.
    pub name: String,
    /// Whether this dependency is wired into the running engine at all.
    pub configured: bool,
    /// The probe outcome.
    pub status: CheckStatus,
    /// A secret-free detail line: the endpoint/deployment on success, or the
    /// error message on failure. `None` when there is nothing useful to add.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl DependencyHealth {
    /// A dependency that is not configured (and therefore not probed).
    pub fn skipped(name: &str) -> Self {
        Self {
            name: name.to_string(),
            configured: false,
            status: CheckStatus::Skipped,
            detail: None,
        }
    }

    /// A configured dependency that passed its probe; `detail` describes what
    /// was reached (e.g. the endpoint or deployment name).
    pub fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            configured: true,
            status: CheckStatus::Ok,
            detail: Some(detail.into()),
        }
    }

    /// A configured dependency whose probe failed; `detail` carries the error.
    pub fn failed(name: &str, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            configured: true,
            status: CheckStatus::Failed,
            detail: Some(detail.into()),
        }
    }
}

/// A full readiness report: the per-dependency checks plus an overall verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HealthReport {
    /// `true` when no *configured* dependency failed its probe. Unconfigured
    /// (skipped) dependencies never make the service unready.
    pub ready: bool,
    /// One entry per dependency, in a stable order.
    pub checks: Vec<DependencyHealth>,
}

impl HealthReport {
    /// Build a report from its checks, deriving `ready` from their statuses.
    ///
    /// The service is considered ready unless at least one configured
    /// dependency reports [`CheckStatus::Failed`].
    pub fn from_checks(checks: Vec<DependencyHealth>) -> Self {
        let ready = !checks
            .iter()
            .any(|check| check.status == CheckStatus::Failed);
        Self { ready, checks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_is_true_when_no_configured_dependency_failed() {
        let report = HealthReport::from_checks(vec![
            DependencyHealth::ok("cosmos", "https://acct/ db=gaia"),
            DependencyHealth::skipped("brave-search"),
        ]);
        assert!(report.ready);
    }

    #[test]
    fn ready_is_false_when_any_configured_dependency_failed() {
        let report = HealthReport::from_checks(vec![
            DependencyHealth::ok("cosmos", "ok"),
            DependencyHealth::failed("foundry-model-router", "401 Unauthorized"),
        ]);
        assert!(!report.ready);
    }

    #[test]
    fn skipped_dependencies_do_not_make_the_service_unready() {
        let report = HealthReport::from_checks(vec![
            DependencyHealth::skipped("cosmos"),
            DependencyHealth::skipped("brave-search"),
        ]);
        assert!(report.ready);
    }

    #[test]
    fn report_serialises_to_the_expected_json_shape() {
        let report = HealthReport::from_checks(vec![
            DependencyHealth::ok("cosmos", "https://acct/ db=gaia"),
            DependencyHealth::failed("brave-search", "403 Forbidden"),
            DependencyHealth::skipped("foundry-embeddings"),
        ]);
        let json = serde_json::to_value(&report).expect("serialises");
        assert_eq!(json["ready"], false);
        assert_eq!(json["checks"][0]["name"], "cosmos");
        assert_eq!(json["checks"][0]["status"], "ok");
        assert_eq!(json["checks"][0]["configured"], true);
        assert_eq!(json["checks"][1]["status"], "failed");
        assert_eq!(json["checks"][1]["detail"], "403 Forbidden");
        assert_eq!(json["checks"][2]["status"], "skipped");
        // `detail` is omitted for the skipped entry.
        assert!(json["checks"][2].get("detail").is_none());
    }
}
