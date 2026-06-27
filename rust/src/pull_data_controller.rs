//! The [`PullDataController`]: Gaia's shared **pull pass** (LLM Call 1 retrieval).
//!
//! This module is the single source of truth for the *retrieval* half of a turn:
//! given the raw text LLM Call 1 produced, it parses the action plan, executes
//! every retrieval action (the Cosmos-backed queries through
//! [`crate::executor::Executor`] and the `Web` action through the Brave
//! [`crate::web_search::BraveClient`]), and folds the results — together with
//! Call 1's analysis/facts/newContext documents — into the eight-section
//! Response Data Context that grounds LLM Call 2.
//!
//! Crucially, both the **live cloud app** ([`crate::engine::Engine`]) and the
//! **data-retrieval self-test** ([`crate::test_data_retrieval`]) drive this same
//! controller. That is the whole point: the test exercises the exact code that
//! runs in production, so the two can never drift apart. The controller makes
//! **no** extra LLM call and never panics; a missing client or an unparsable
//! plan simply yields empty sections plus an explanatory note.
//!
//! The controller returns a rich [`PullResult`] so each caller can take what it
//! needs: the engine uses only [`PullResult::context`] and
//! [`PullResult::searches`]; the self-test additionally reports the per-action
//! [`PullResult::notes`], [`PullResult::planned_queries`], and the
//! [`PullResult::all_ok`] pass/fail gate.

use serde::Serialize;

use crate::actions::{ActionPlan, ActionsFile, SessionContext};
use crate::cosmos::CosmosClient;
use crate::embeddings::EmbeddingClient;
use crate::executor::Executor;
use crate::response_context::{build_response_data_context, parse_call1_extras, RetrievalGroup};
use crate::web_search::BraveClient;

/// The exact Cosmos query that was planned for one retrieval action.
///
/// This makes the retrieval mode auditable: a `VectorDistance(...)` SQL proves
/// semantic search ran, while a `CONTAINS(...)` SQL proves keyword search ran —
/// independent of what the raw model reply originally authored. The self-test
/// serializes these to `queries.json`; the cloud app ignores them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryAudit {
    /// The action id (e.g. `q3`).
    pub id: String,
    /// The target container (e.g. `GaiaKB`).
    pub target: String,
    /// The resolved retrieval mode label (`semantic` or `keyword`).
    pub mode: String,
    /// The single logical partition the query is pinned to.
    pub partition_value: String,
    /// The exact parameterised Cosmos SQL sent to the account.
    pub sql: String,
}

/// Everything the pull pass produced for one turn.
///
/// Callers take only the fields they need. The engine reads `context` and
/// `searches`; the self-test additionally reads `plan`, `groups`,
/// `planned_queries`, `notes`, the action counts, and `all_ok`.
pub struct PullResult {
    /// The parsed action plan (after any force-semantic rewrite), or `None`
    /// when Call 1's reply contained no parseable `actions.json` document.
    pub plan: Option<ActionsFile>,
    /// Per-action retrieval results (Cosmos records and Brave results, each
    /// serialized to generic JSON values), in execution order. Call 1's
    /// non-action documents (analysis, facts, newContext summary) are folded
    /// into `context` during construction, so they are not retained separately.
    pub groups: Vec<RetrievalGroup>,
    /// The web queries that were actually issued to Brave this turn.
    pub searches: Vec<String>,
    /// The exact SQL planned per Cosmos action. Empty unless `capture_queries`
    /// was requested (capturing semantic queries costs an extra embedding call).
    pub planned_queries: Vec<QueryAudit>,
    /// The fully-assembled, eight-section Response Data Context markdown.
    pub context: String,
    /// How many Cosmos-backed (non-`Web`) actions were attempted.
    pub cosmos_actions: usize,
    /// How many `Web` actions were attempted.
    pub web_actions: usize,
    /// Human-readable notes explaining any failed or skipped retrieval.
    pub notes: Vec<String>,
    /// `true` only when every attempted retrieval succeeded. The cloud app
    /// ignores this; the self-test turns it into a pass/fail verdict.
    pub all_ok: bool,
}

/// Runs the pull pass. Holds borrowed references to the (optional) retrieval
/// clients; cheap to construct per turn from the caller's owned clients.
pub struct PullDataController<'a> {
    /// The Cosmos client, or `None` when Cosmos is not configured.
    cosmos: Option<&'a CosmosClient>,
    /// The embedding client used when an action chooses semantic mode, or
    /// `None` when embeddings are not configured.
    embedder: Option<&'a EmbeddingClient>,
    /// The Brave web-search client, or `None` when web search is not configured.
    web: Option<&'a BraveClient>,
}

impl<'a> PullDataController<'a> {
    /// Build a controller from borrowed retrieval clients.
    pub fn new(
        cosmos: Option<&'a CosmosClient>,
        embedder: Option<&'a EmbeddingClient>,
        web: Option<&'a BraveClient>,
    ) -> Self {
        Self {
            cosmos,
            embedder,
            web,
        }
    }

    /// Execute the pull pass for one turn.
    ///
    /// Parses `call1_raw` (the raw LLM Call 1 reply), runs every retrieval
    /// action it planned, and assembles the Response Data Context. When
    /// `capture_queries` is `true` the controller also records the exact SQL of
    /// each Cosmos query into [`PullResult::planned_queries`] — useful for the
    /// self-test's audit artifacts, but it costs an extra embedding call per
    /// semantic action, so the cloud app passes `false`.
    ///
    /// `input` is the user's message (used as the web-search fallback and folded
    /// into the context). This method is infallible: any error is captured as a
    /// note and reflected in [`PullResult::all_ok`] rather than returned.
    pub fn execute(
        &self,
        user_id: &str,
        input: &str,
        requested_at: &str,
        call1_raw: &str,
        capture_queries: bool,
    ) -> PullResult {
        // Call 1's non-action documents always parse (degrading to defaults).
        let extras = parse_call1_extras(call1_raw);

        let mut groups: Vec<RetrievalGroup> = Vec::new();
        let mut searches: Vec<String> = Vec::new();
        let mut planned_queries: Vec<QueryAudit> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut all_ok = true;
        let mut cosmos_action_count = 0;
        let mut web_action_count = 0;

        // Parse the action plan. Without one we still emit a deterministic (but
        // result-free) context from the extras alone.
        let plan = crate::actions::parse_call1_actions(call1_raw).map(|mut actions| {
            // Honour the GAIA_FORCE_SEMANTIC override before anything reads the
            // plan, so both the captured artifact and the executed query reflect
            // the override.
            if crate::executor::force_semantic() {
                crate::executor::force_semantic_on(&mut actions);
            }
            actions
        });

        if let Some(actions) = &plan {
            // Split the plan into Cosmos-backed queries and Web searches.
            let (web_actions, cosmos_actions): (Vec<ActionPlan>, Vec<ActionPlan>) = actions
                .actions
                .iter()
                .cloned()
                .partition(|action| action.target.eq_ignore_ascii_case("Web"));

            cosmos_action_count = cosmos_actions.len();
            web_action_count = web_actions.len();

            self.run_cosmos(
                user_id,
                requested_at,
                &actions.version,
                cosmos_actions,
                capture_queries,
                &mut groups,
                &mut planned_queries,
                &mut notes,
                &mut all_ok,
            );

            self.run_web(
                input,
                web_actions,
                &mut groups,
                &mut searches,
                &mut notes,
                &mut all_ok,
            );
        }

        let context = build_response_data_context(user_id, input, requested_at, &extras, &groups);

        PullResult {
            plan,
            groups,
            searches,
            planned_queries,
            context,
            cosmos_actions: cosmos_action_count,
            web_actions: web_action_count,
            notes,
            all_ok,
        }
    }

    /// Execute the Cosmos-backed (non-`Web`) actions, appending their results to
    /// `groups` and any failures to `notes`.
    #[allow(clippy::too_many_arguments)] // Threading the shared accumulators keeps `execute` readable.
    fn run_cosmos(
        &self,
        user_id: &str,
        requested_at: &str,
        version: &str,
        cosmos_actions: Vec<ActionPlan>,
        capture_queries: bool,
        groups: &mut Vec<RetrievalGroup>,
        planned_queries: &mut Vec<QueryAudit>,
        notes: &mut Vec<String>,
        all_ok: &mut bool,
    ) {
        if cosmos_actions.is_empty() {
            return;
        }

        let Some(client) = self.cosmos else {
            *all_ok = false;
            notes.push(format!(
                "{} Cosmos action(s) requested but Cosmos is not configured",
                cosmos_actions.len()
            ));
            return;
        };

        // Preserve the action id + target alongside each outcome.
        let targets: Vec<String> = cosmos_actions.iter().map(|a| a.target.clone()).collect();
        let action_ids: Vec<String> = cosmos_actions.iter().map(|a| a.id.clone()).collect();
        let plan = ActionsFile {
            version: version.to_string(),
            session: SessionContext {
                user_id: user_id.to_string(),
                requested_at: requested_at.to_string(),
            },
            actions: cosmos_actions,
        };

        // Capture the exact SQL each query will run, so the artifact is
        // auditable. This re-plans the query (binding @queryVector for semantic
        // actions), which is why it is gated behind `capture_queries`.
        if capture_queries {
            for action in &plan.actions {
                if let Ok(planned) = crate::executor::plan_for(action, self.embedder) {
                    let mode = if planned.sql.contains("VectorDistance") {
                        "semantic"
                    } else {
                        "keyword"
                    };
                    planned_queries.push(QueryAudit {
                        id: action.id.clone(),
                        target: action.target.clone(),
                        mode: mode.to_string(),
                        partition_value: planned.partition_value,
                        sql: planned.sql,
                    });
                }
            }
        }

        let outcomes = match self.embedder {
            Some(embedder) => Executor::with_embedder(client, Some(embedder.clone())).run(&plan),
            None => Executor::new(client).run(&plan),
        };
        for ((target, action_id), outcome) in
            targets.iter().zip(action_ids.iter()).zip(outcomes.iter())
        {
            match &outcome.result {
                Ok(records) => {
                    let values: Vec<serde_json::Value> = records
                        .iter()
                        .filter_map(|r| serde_json::to_value(r).ok())
                        .collect();
                    groups.push(RetrievalGroup {
                        action_id: action_id.clone(),
                        container: target.clone(),
                        records: values,
                    });
                }
                Err(err) => {
                    *all_ok = false;
                    notes.push(format!("Cosmos action {} failed: {err}", outcome.id));
                }
            }
        }
    }

    /// Execute the `Web` actions against Brave, appending their results to
    /// `groups`, the issued queries to `searches`, and any failures to `notes`.
    fn run_web(
        &self,
        input: &str,
        web_actions: Vec<ActionPlan>,
        groups: &mut Vec<RetrievalGroup>,
        searches: &mut Vec<String>,
        notes: &mut Vec<String>,
        all_ok: &mut bool,
    ) {
        if web_actions.is_empty() {
            return;
        }

        let Some(client) = self.web else {
            *all_ok = false;
            notes.push(format!(
                "{} Web action(s) requested but Brave is not configured",
                web_actions.len()
            ));
            return;
        };

        for action in &web_actions {
            if let Err(err) = action.validate() {
                *all_ok = false;
                notes.push(format!("Web action {} failed validation: {err}", action.id));
                continue;
            }

            let query = web_query_for(action, input);
            match client.search(&query, action.effective_top()) {
                Ok(results) => {
                    searches.push(query);
                    let values: Vec<serde_json::Value> = results
                        .iter()
                        .filter_map(|r| serde_json::to_value(r).ok())
                        .collect();
                    groups.push(RetrievalGroup {
                        action_id: action.id.clone(),
                        container: "Web".to_string(),
                        records: values,
                    });
                }
                Err(err) => {
                    *all_ok = false;
                    notes.push(format!("Brave action {} failed: {err}", action.id));
                }
            }
        }
    }
}

/// Choose the text to search the web with for a `Web` action.
///
/// The model's free-text `text` filter is the most precise signal, then its
/// `intent`; if neither is present we fall back to `fallback` (the user's
/// message) so a `Web` action always has *something* to search for. Both the
/// cloud app and the self-test call this, so they behave identically.
pub fn web_query_for(action: &ActionPlan, fallback: &str) -> String {
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
    fallback.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ActionFilters;

    /// Build a minimal `Web` action for the pure-helper tests.
    fn web_action(intent: &str, text: Option<&str>) -> ActionPlan {
        ActionPlan {
            id: "q1".to_string(),
            kind: "search".to_string(),
            target: "Web".to_string(),
            user_id: Some("threadkeeper".to_string()),
            entity: Some("threadkeeper".to_string()),
            intent: intent.to_string(),
            top: 3,
            query: None,
            filters: ActionFilters {
                text: text.map(str::to_string),
                ..ActionFilters::default()
            },
        }
    }

    #[test]
    fn web_query_prefers_text_then_intent_then_fallback() {
        // The free-text filter wins when present.
        let with_text = web_action("broad intent", Some("  mars rovers  "));
        assert_eq!(web_query_for(&with_text, "the question"), "mars rovers");

        // Otherwise the intent is used.
        let with_intent = web_action("latest mars news", None);
        assert_eq!(
            web_query_for(&with_intent, "the question"),
            "latest mars news"
        );

        // With neither, fall back to the caller's input.
        let bare = web_action("   ", None);
        assert_eq!(web_query_for(&bare, "  the question  "), "the question");
    }

    #[test]
    fn execute_without_clients_degrades_to_an_empty_but_complete_context() {
        // No Cosmos/embedder/Brave clients: every retrieval section must still
        // be emitted, Call 1's extras folded in, and the requested retrieval
        // recorded as failed (a needed-but-missing client).
        let controller = PullDataController::new(None, None, None);
        let reply = r#"[
          { "version": "1.0",
            "session": { "user_id": "alice", "requested_at": "2026-06-21T00:00:00Z" },
            "actions": [ { "id": "q1", "kind": "search", "target": "Web", "intent": "mars news", "top": 3, "filters": {} } ] },
          { "emotion": "calm", "truthfulness": "honest", "intention": "learn" },
          [ { "fact": "favourite_colour", "value": "blue" } ],
          { "summary": "We spoke about colours before." }
        ]"#;

        let result = controller.execute("alice", "hi", "2026-06-21T00:00:00Z", reply, false);

        // The plan parsed and was split correctly.
        assert!(result.plan.is_some());
        assert_eq!(result.web_actions, 1);
        assert_eq!(result.cosmos_actions, 0);
        // The Web client was missing, so the attempt failed.
        assert!(!result.all_ok);
        assert!(result
            .notes
            .iter()
            .any(|n| n.contains("Brave is not configured")));
        // No results were gathered, but the context is still complete.
        assert!(result.groups.is_empty());
        assert!(result.searches.is_empty());
        for heading in [
            "## WebSearchResults",
            "## DataLakeResults",
            "## KnowledgeBaseResults",
            "## ConnectionsResults",
            "## EmotionResults",
            "## TruthfulNessResults",
            "## IntentionResults",
            "## OldContextSummary",
        ] {
            assert!(
                result.context.contains(heading),
                "missing heading {heading}"
            );
        }
        // Call 1's extras are folded into their sections.
        assert!(result.context.contains("calm"));
        assert!(result.context.contains("We spoke about colours before."));
        assert!(result.context.contains("**favourite_colour:** blue"));

        // Determinism: identical inputs produce identical context.
        let again = controller.execute("alice", "hi", "2026-06-21T00:00:00Z", reply, false);
        assert_eq!(result.context, again.context);
    }

    #[test]
    fn execute_without_a_parseable_plan_still_builds_a_context() {
        // A reply with no JSON array yields no plan, but the context (built from
        // empty extras) is still complete and nothing is reported as failed.
        let controller = PullDataController::new(None, None, None);
        let result = controller.execute("alice", "hi", "2026-06-21T00:00:00Z", "not json", false);
        assert!(result.plan.is_none());
        assert_eq!(result.cosmos_actions, 0);
        assert_eq!(result.web_actions, 0);
        assert!(result.all_ok);
        assert!(result.notes.is_empty());
        assert!(result.context.contains("## WebSearchResults"));
    }
}
