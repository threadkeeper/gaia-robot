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

/// Minimum number of results to fetch for a `Web` action.
///
/// Web search snippets are noisy: the single top-ranked page is often a
/// JavaScript-rendered landing page (e.g. a weather site) whose description is a
/// run-on of forecast phrases with no usable numbers, while the concrete facts
/// (a temperature, a "no rain expected") live in results #2–#5. The model's
/// `top` is tuned for Cosmos top-N reads, so for `Web` actions we ignore a tiny
/// `top` and always request at least this many results to give LLM Call 2 enough
/// grounding to actually answer. A larger model-chosen `top` is still honoured.
const MIN_WEB_RESULT_COUNT: usize = 5;

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

/// One retrieval action's label and the wall-clock time it took.
///
/// Mirrors the push pass's `PushActionTiming` so the UI debug panel can render
/// every action — Cosmos reads (`q3 → GaiaKB`) and web searches
/// (`q1 → Web`) — with the milliseconds it cost. `ms` may be fractional for a
/// sub-millisecond read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PullActionTiming {
    /// A short label for the action, e.g. `q3 → GaiaKB` or `q1 → Web`.
    #[serde(rename = "type")]
    pub action_type: String,
    /// Milliseconds spent executing this one retrieval (may be fractional).
    pub ms: f64,
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
    /// Per-action label + wall-clock milliseconds for every retrieval that ran
    /// this turn (Cosmos reads and web searches), in plan order. The cloud app
    /// surfaces these in the debug panel so each action shows the time it cost.
    pub action_timings: Vec<PullActionTiming>,
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
        let mut action_timings: Vec<PullActionTiming> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut all_ok = true;
        let mut cosmos_action_count = 0;
        let mut web_action_count = 0;

        // Parse the action plan, defaulting to an empty plan so the guaranteed
        // core-user queries below still run even when Call 1 authored nothing
        // parseable (a missing/garbled reply must never leave Gaia blind to who
        // the user is).
        let mut actions = crate::actions::parse_call1_actions(call1_raw).unwrap_or_default();

        // GUARANTEE Gaia always retrieves her core knowledge about the user this
        // turn — her KB facts, diary reflections, and the connection ledger —
        // regardless of what Call 1 planned, so she never "forgets who you are".
        // Done before the force-semantic override so that override upgrades the
        // embeddable core queries too.
        crate::executor::ensure_core_user_queries(&mut actions, user_id);

        // Honour the GAIA_FORCE_SEMANTIC override before anything reads the
        // plan, so both the captured artifact and the executed query reflect
        // the override.
        if crate::executor::force_semantic() {
            crate::executor::force_semantic_on(&mut actions);
        }

        let plan = Some(actions);

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
                &mut action_timings,
                &mut notes,
                &mut all_ok,
            );

            self.run_web(
                input,
                web_actions,
                &mut groups,
                &mut searches,
                &mut action_timings,
                &mut notes,
                &mut all_ok,
            );
        }

        // Re-order the timings into the original plan order so the debug panel
        // lists them exactly as Call 1 authored them, even though Cosmos reads
        // and web searches were executed in separate batches above.
        if let Some(actions) = &plan {
            action_timings = order_timings_by_plan(actions, action_timings);
        }

        let context = build_response_data_context(user_id, input, requested_at, &extras, &groups);

        PullResult {
            plan,
            groups,
            searches,
            planned_queries,
            action_timings,
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
        action_timings: &mut Vec<PullActionTiming>,
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
            // Still list every planned action (with 0 ms) so the debug panel
            // shows what Call 1 intended to read even though no read ran.
            for action in &cosmos_actions {
                action_timings.push(PullActionTiming {
                    action_type: format!("{} → {}", action.id, action.target),
                    ms: 0.0,
                });
            }
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
            // Record the per-action read time regardless of success/failure so
            // the debug panel always shows what each retrieval cost.
            action_timings.push(PullActionTiming {
                action_type: format!("{action_id} → {target}"),
                ms: outcome.ms,
            });
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
    #[allow(clippy::too_many_arguments)] // Threading the shared accumulators keeps `execute` readable.
    fn run_web(
        &self,
        input: &str,
        web_actions: Vec<ActionPlan>,
        groups: &mut Vec<RetrievalGroup>,
        searches: &mut Vec<String>,
        action_timings: &mut Vec<PullActionTiming>,
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
            // Still list every planned web action (with 0 ms) so the debug panel
            // shows what Call 1 intended to search even though no search ran.
            for action in &web_actions {
                action_timings.push(PullActionTiming {
                    action_type: format!("{} → Web", action.id),
                    ms: 0.0,
                });
            }
            return;
        };

        for action in &web_actions {
            if let Err(err) = action.validate() {
                *all_ok = false;
                notes.push(format!("Web action {} failed validation: {err}", action.id));
                continue;
            }

            let query = web_query_for(action, input);
            // Fetch at least MIN_WEB_RESULT_COUNT results so a noisy top snippet
            // does not starve the context; honour a larger model-chosen top.
            // Time the call so the web search reports its latency like the
            // Cosmos reads do.
            let start = std::time::Instant::now();
            let outcome = client.search(&query, web_result_count(action));
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            action_timings.push(PullActionTiming {
                action_type: format!("{} → Web", action.id),
                ms,
            });
            match outcome {
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

/// Re-order the collected per-action timings to match the plan's action order.
///
/// `run_cosmos` and `run_web` execute in two separate batches (Cosmos reads
/// first, then web searches), so the timings are gathered grouped by kind. This
/// restores the original Call 1 order by walking the plan and pulling each
/// action's timing out by its `id → target` label, so the debug panel lists the
/// actions exactly as the model authored them. Any timing whose label does not
/// match a plan action (there should be none) is appended at the end so nothing
/// is silently dropped.
fn order_timings_by_plan(
    plan: &ActionsFile,
    mut timings: Vec<PullActionTiming>,
) -> Vec<PullActionTiming> {
    let mut ordered = Vec::with_capacity(timings.len());
    for action in &plan.actions {
        let label = format!("{} → {}", action.id, action.target);
        if let Some(pos) = timings.iter().position(|t| t.action_type == label) {
            ordered.push(timings.remove(pos));
        }
    }
    // Preserve any unmatched timings rather than losing them.
    ordered.append(&mut timings);
    ordered
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

/// Decide how many results to request from Brave for one `Web` action.
///
/// The model's `top` is tuned for Cosmos top-N reads and is frequently `1` for a
/// "give me the current X" web question. A single web snippet is unreliable
/// grounding — the top-ranked page is often a script-rendered landing page whose
/// description carries no usable facts — so we request at least
/// [`MIN_WEB_RESULT_COUNT`] results, while still honouring a larger model-chosen
/// `top`.
fn web_result_count(action: &ActionPlan) -> usize {
    action.effective_top().max(MIN_WEB_RESULT_COUNT)
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
    fn web_result_count_enforces_a_minimum_floor() {
        // A tiny model-chosen top (the common "give me the current X" case) is
        // raised to the floor so a single noisy snippet cannot starve grounding.
        let mut action = web_action("current weather", None);
        action.top = 1;
        assert_eq!(web_result_count(&action), MIN_WEB_RESULT_COUNT);

        // top = 0 means "unset"; effective_top() makes it 3, still below the floor.
        action.top = 0;
        assert_eq!(web_result_count(&action), MIN_WEB_RESULT_COUNT);
    }

    #[test]
    fn web_result_count_honours_a_larger_model_choice() {
        // When the model deliberately asks for more than the floor, respect it.
        let mut action = web_action("broad survey", None);
        action.top = 12;
        assert_eq!(web_result_count(&action), 12);
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
        // The three guaranteed core-user queries (KB, diary, connections) are
        // always appended, so the model's Web-only plan still drives 3 Cosmos
        // reads.
        assert_eq!(result.cosmos_actions, 3);
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
    fn execute_without_a_parseable_plan_still_forces_core_user_queries() {
        // A reply with no JSON array yields no model plan, but Gaia must still
        // never "forget who you are": the three guaranteed core-user queries
        // (KB, diary, connections) are appended so the turn still tries to read
        // the user's identity context. With no Cosmos client they fail, but the
        // context (built from empty extras) is still complete.
        let controller = PullDataController::new(None, None, None);
        let result = controller.execute("alice", "hi", "2026-06-21T00:00:00Z", "not json", false);
        assert!(result.plan.is_some());
        // The three core-user containers are always queried.
        let plan = result.plan.as_ref().unwrap();
        assert_eq!(plan.actions.len(), 3);
        for container in ["GaiaKB", "GaiaDiary", "GaiaConnections"] {
            assert!(
                plan.actions
                    .iter()
                    .any(|a| a.target == container && a.entity.as_deref() == Some("alice")),
                "missing forced core query for {container}"
            );
        }
        assert_eq!(result.cosmos_actions, 3);
        assert_eq!(result.web_actions, 0);
        // Cosmos was not configured, so the forced reads are reported as failed.
        assert!(!result.all_ok);
        assert!(result
            .notes
            .iter()
            .any(|n| n.contains("Cosmos is not configured")));
        assert!(result.context.contains("## WebSearchResults"));
    }

    #[test]
    fn execute_records_a_timing_for_every_planned_action_in_plan_order() {
        // Even with no clients (so nothing actually runs), every planned action
        // must appear in `action_timings` so the debug panel can attach a ms to
        // each one. The Web action is authored first, then the three forced
        // core-user Cosmos reads, and the timings must follow that plan order.
        let controller = PullDataController::new(None, None, None);
        let reply = r#"[
          { "version": "1.0",
            "session": { "user_id": "alice", "requested_at": "2026-06-21T00:00:00Z" },
            "actions": [ { "id": "q1", "kind": "search", "target": "Web", "intent": "mars news", "top": 3, "filters": {} } ] },
          {}, [], {}
        ]"#;

        let result = controller.execute("alice", "hi", "2026-06-21T00:00:00Z", reply, false);

        let labels: Vec<&str> = result
            .action_timings
            .iter()
            .map(|t| t.action_type.as_str())
            .collect();
        // Web first (plan order), then the three forced core-user reads.
        assert_eq!(
            labels,
            vec![
                "q1 → Web",
                "core-gaiakb → GaiaKB",
                "core-gaiadiary → GaiaDiary",
                "core-gaiaconnections → GaiaConnections",
            ]
        );
        // No client ran, so each timing is a deterministic 0 ms placeholder.
        assert!(result.action_timings.iter().all(|t| t.ms == 0.0));
    }

    #[test]
    fn order_timings_by_plan_restores_authored_order() {
        // Timings gathered Cosmos-first then web are re-sorted into the order the
        // model authored the actions, and unmatched timings are kept at the end.
        let plan = ActionsFile {
            version: "1.0".to_string(),
            session: SessionContext::default(),
            actions: vec![
                web_action("mars", None), // id q1, target Web
                ActionPlan {
                    id: "q2".to_string(),
                    kind: "query".to_string(),
                    target: "GaiaKB".to_string(),
                    user_id: Some("alice".to_string()),
                    entity: Some("alice".to_string()),
                    intent: "kb".to_string(),
                    top: 3,
                    query: None,
                    filters: ActionFilters::default(),
                },
            ],
        };
        // Gathered out of order: Cosmos (q2) before web (q1), plus a stray.
        let gathered = vec![
            PullActionTiming {
                action_type: "q2 → GaiaKB".to_string(),
                ms: 5.0,
            },
            PullActionTiming {
                action_type: "q1 → Web".to_string(),
                ms: 9.0,
            },
            PullActionTiming {
                action_type: "qX → Unknown".to_string(),
                ms: 1.0,
            },
        ];

        let ordered = order_timings_by_plan(&plan, gathered);
        let labels: Vec<&str> = ordered.iter().map(|t| t.action_type.as_str()).collect();
        // Plan order first (q1 then q2), then any unmatched timing.
        assert_eq!(labels, vec!["q1 → Web", "q2 → GaiaKB", "qX → Unknown"]);
    }

    #[test]
    fn execute_runs_cosmos_and_web_actions_against_their_clients() {
        // Drive the full pull pass with live (mock) Cosmos and Brave clients: the
        // authored UsersKB keyword query and the Web search both succeed, and the
        // three guaranteed core-user queries (GaiaKB, GaiaDiary, GaiaConnections)
        // also run (returning nothing here). Their records are folded into the
        // context and the issued web query is recorded.
        let kb_doc = r#"{"Documents":[
            {"id":"UsersKB|alice|2026-05-10","userId":"alice","date":"2026-05-10","data":"prefers tea"}
        ]}"#;
        let empty = r#"{"Documents":[]}"#;
        // One response per Cosmos call, in plan order: the authored q1 (UsersKB)
        // first, then the three forced core-user queries.
        let (cosmos_endpoint, cosmos_handle) = crate::test_http::spawn_mock_http_sequence(vec![
            ("200 OK".to_string(), kb_doc.to_string()),
            ("200 OK".to_string(), empty.to_string()),
            ("200 OK".to_string(), empty.to_string()),
            ("200 OK".to_string(), empty.to_string()),
        ]);
        let cosmos = CosmosClient::new(cosmos_endpoint, "gaia", "tok");

        let brave_body = r#"{"web":{"results":[
            {"title":"Mars","url":"https://example.com/mars","description":"red planet"}
        ]}}"#;
        let (brave_endpoint, brave_handle) =
            crate::test_http::spawn_mock_http("200 OK", brave_body);
        let brave = crate::web_search::BraveClient::for_test(brave_endpoint);

        let controller = PullDataController::new(Some(&cosmos), None, Some(&brave));
        let reply = r#"[
          { "version": "1.0",
            "session": { "user_id": "alice", "requested_at": "2026-06-21T00:00:00Z" },
            "actions": [
              { "id": "q1", "kind": "query", "target": "UsersKB", "user_id": "alice", "entity": "alice", "intent": "past notes", "top": 3, "filters": {} },
              { "id": "q2", "kind": "query", "target": "Web", "user_id": "alice", "entity": "alice", "intent": "mars news", "top": 3, "filters": {} }
            ] }
        ]"#;

        let result = controller.execute("alice", "mars?", "2026-06-21T00:00:00Z", reply, false);

        // The authored UsersKB query plus the three forced core-user queries.
        assert_eq!(result.cosmos_actions, 4);
        assert_eq!(result.web_actions, 1);
        assert!(result.all_ok, "notes: {:?}", result.notes);
        // Four Cosmos groups (one per query) plus the one web group.
        assert_eq!(result.groups.len(), 5);
        assert_eq!(result.searches, vec!["mars news".to_string()]);
        assert!(result.context.contains("prefers tea"));
        assert!(result.context.contains("Mars"));

        cosmos_handle.join().expect("cosmos mock thread joins");
        brave_handle.join().expect("brave mock thread joins");
    }

    #[test]
    fn execute_notes_a_cosmos_failure_but_still_completes() {
        // Cosmos returns an error status for every read (the authored UsersKB
        // query and the three forced core-user queries): each is recorded as
        // failed and all_ok flips, yet the context is still fully assembled.
        let denied = (
            "403 Forbidden".to_string(),
            r#"{"message":"denied"}"#.to_string(),
        );
        let (cosmos_endpoint, cosmos_handle) = crate::test_http::spawn_mock_http_sequence(vec![
            denied.clone(),
            denied.clone(),
            denied.clone(),
            denied,
        ]);
        let cosmos = CosmosClient::new(cosmos_endpoint, "gaia", "tok");

        let controller = PullDataController::new(Some(&cosmos), None, None);
        let reply = r#"[
          { "version": "1.0",
            "session": { "user_id": "alice", "requested_at": "2026-06-21T00:00:00Z" },
            "actions": [
              { "id": "q1", "kind": "query", "target": "UsersKB", "user_id": "alice", "entity": "alice", "intent": "past notes", "top": 3, "filters": {} }
            ] }
        ]"#;

        let result = controller.execute("alice", "hi", "2026-06-21T00:00:00Z", reply, false);

        assert_eq!(result.cosmos_actions, 4);
        assert!(!result.all_ok);
        assert!(result.notes.iter().any(|n| n.contains("failed")));
        assert!(result.context.contains("## DataLakeResults"));
        cosmos_handle.join().expect("cosmos mock thread joins");
    }
}
