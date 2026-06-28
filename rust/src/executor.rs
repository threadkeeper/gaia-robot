#![allow(dead_code)]

//! The [`Executor`] type: turns an `actions.json` plan into Cosmos queries.
//!
//! The first LLM pass emits an [`ActionsFile`] (see [`crate::actions`]). The
//! executor walks each [`ActionPlan`], translates it into a safe, parameterised
//! Cosmos SQL query, runs it through a [`CosmosClient`], and collects the
//! retrieved [`Record`]s. This is the **read** half of wiring Cosmos into the
//! program flow; writes go straight through [`CosmosClient::upsert`].
//!
//! Query construction supports two retrieval modes selected per action:
//! - `keyword` (`CONTAINS(LOWER(...))`)
//! - `semantic` (`VectorDistance(...)` with a query-time embedding)
//!
//! User-controlled values are always passed as [`QueryParam`]s (never
//! string-interpolated) to avoid query injection; only program-owned constants
//! such as `TOP` are interpolated.

use crate::actions::{ActionFilters, ActionPlan, ActionsFile};
use crate::cosmos::{CosmosClient, QueryParam};
use crate::embeddings::EmbeddingClient;
use crate::storage::Record;

/// Native Cosmos vector-search options for DiskANN-backed retrieval.
///
/// - `bool_expr = false` in `VectorDistance` tells Cosmos to use the index.
/// - `searchListSizeMultiplier` trades a little RU/latency for better recall.
/// - `filterPriority` balances vector ranking against the WHERE filter.
const VECTOR_DISTANCE_OPTIONS: &str =
    "{distanceFunction:'Cosine',dataType:'Float32',searchListSizeMultiplier:10,filterPriority:0.75}";

/// The outcome of executing one [`ActionPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct ActionOutcome {
    /// The originating action's id (e.g. `q1`).
    pub id: String,
    /// The retrieved records, or an error message describing what failed.
    pub result: Result<Vec<Record>, String>,
    /// Wall-clock milliseconds this single action's Cosmos read took (may be
    /// fractional). Surfaced in the UI debug panel so every retrieval shows the
    /// time it cost, alongside the push-pass action timings.
    pub ms: f64,
}

/// A planned, ready-to-run Cosmos query derived from an [`ActionPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedQuery {
    /// The logical partition value (entity/userId) scoping the query.
    pub partition_value: String,
    /// The parameterised Cosmos SQL text.
    pub sql: String,
    /// The bound parameters referenced by `sql`.
    pub params: Vec<QueryParam>,
}

/// Runs action plans against a Cosmos account.
#[derive(Debug, Clone)]
pub struct Executor<'a> {
    client: &'a CosmosClient,
    embedder: Option<EmbeddingClient>,
}

impl<'a> Executor<'a> {
    /// Create an executor that runs against `client`.
    pub fn new(client: &'a CosmosClient) -> Self {
        Self {
            client,
            embedder: None,
        }
    }

    /// Create an executor that can run semantic queries with `embedder`.
    pub fn with_embedder(client: &'a CosmosClient, embedder: Option<EmbeddingClient>) -> Self {
        Self { client, embedder }
    }

    /// Execute every action in the plan, returning one outcome per action.
    ///
    /// One failing action never aborts the others; its error is captured in its
    /// own [`ActionOutcome`].
    pub fn run(&self, actions: &ActionsFile) -> Vec<ActionOutcome> {
        actions
            .actions
            .iter()
            .map(|action| {
                // Time each action individually so the debug panel can attribute
                // the millisecond cost to exactly one retrieval.
                let start = std::time::Instant::now();
                let result = self.run_one(action);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                ActionOutcome {
                    id: action.id.clone(),
                    result,
                    ms,
                }
            })
            .collect()
    }

    /// Validate, plan, and execute a single action.
    fn run_one(&self, action: &ActionPlan) -> Result<Vec<Record>, String> {
        action.validate().map_err(|err| err.to_string())?;
        let planned = plan_for(action, self.embedder.as_ref())?;
        let mut records = self
            .client
            .query(
                &action.target,
                &planned.partition_value,
                &planned.sql,
                &planned.params,
            )
            .map_err(|err| err.to_string())?;

        // Defence in depth for "keep the payload small for LLM Call 2": even if
        // the authored query forgot its `TOP n`, never hand more than the
        // requested number of records back into the Response Data Context.
        records.truncate(action.effective_top());
        Ok(records)
    }
}

/// Choose the query to run for an action: the LLM-authored one when present and
/// safe, otherwise a query built from the structured fields.
///
/// When the model authored a [`ActionPlan::query`], we prefer it (the model has
/// the full container schema and can express keyword or vector search precisely)
/// but only after [`validate_read_only_query`] confirms it is a single read-only
/// `SELECT`. Either way the query runs against exactly one logical partition:
/// [`partition_for`] decides the partition value and the Cosmos client pins it
/// with the partition-key header, so a query can never read another user's or
/// entity's data.
pub fn plan_for(
    action: &ActionPlan,
    embedder: Option<&EmbeddingClient>,
) -> Result<PlannedQuery, String> {
    match action.authored_query() {
        Some(sql) => {
            validate_read_only_query(sql)?;
            let (_field, partition_value) = partition_for(action)?;

            // Bind the partition value only when the query references `@pk`, so
            // the model can write `c.<key> = @pk` without inlining the value.
            let mut params = Vec::new();
            if sql.contains("@pk") {
                params.push(QueryParam::new("@pk", partition_value.clone()));
            }

            // If the authored SQL asks for a query vector placeholder, bind it
            // from the action's semantic text at runtime.
            if sql.contains("@queryVector") {
                if !target_supports_semantic(&action.target) {
                    return Err(format!(
                        "action '{}' targets {} but semantic mode is not supported there",
                        action.id, action.target
                    ));
                }
                let semantic_text = semantic_query_text(action).ok_or_else(|| {
                    format!(
                        "action '{}' authored @queryVector but provided no semantic/text/intent input",
                        action.id
                    )
                })?;
                let query_vector = embed_query(embedder, semantic_text)?;
                params.push(query_vector_param(query_vector));
            }

            Ok(PlannedQuery {
                partition_value,
                sql: sql.to_string(),
                params,
            })
        }
        None => plan_to_query(action, embedder),
    }
}

/// Reject anything that is not a single, read-only `SELECT` statement.
///
/// The Cosmos query API cannot mutate data, and the client always pins a single
/// partition, so the blast radius is already tiny. This is defence in depth: it
/// guarantees the model handed us *one* `SELECT` (no statement chaining) before
/// we send it, and gives a clear error rather than a confusing Cosmos 400 if the
/// model ever emits something else.
fn validate_read_only_query(sql: &str) -> Result<(), String> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err("authored query is empty".to_string());
    }
    // A single statement only: Cosmos SQL never needs a `;`, and forbidding it
    // rules out trailing-statement injection outright.
    if trimmed.contains(';') {
        return Err("authored query must be a single statement (no ';')".to_string());
    }
    // Must be a projection query. `SELECT` is the only read shape Cosmos exposes.
    let head: String = trimmed
        .chars()
        .take(6)
        .flat_map(char::to_lowercase)
        .collect();
    if head != "select" {
        return Err("authored query must start with SELECT".to_string());
    }
    Ok(())
}

/// Translate an [`ActionPlan`] into a parameterised, single-partition query.
///
/// The partition field follows the container family: `Users*` containers
/// partition on `userId` (taken from `action.user_id`), everything else on
/// `entity` (taken from `action.entity`). Optional date-range filters are
/// appended in both retrieval modes. Text filters are used for keyword mode,
/// while semantic mode binds a query vector and orders by `VectorDistance`.
pub fn plan_to_query(
    action: &ActionPlan,
    embedder: Option<&EmbeddingClient>,
) -> Result<PlannedQuery, String> {
    match mode_for(action)? {
        RetrievalMode::Keyword => plan_to_keyword_query(action),
        RetrievalMode::Semantic => {
            if !target_supports_semantic(&action.target) {
                return Err(format!(
                    "action '{}' targets {} but semantic mode is not supported there",
                    action.id, action.target
                ));
            }
            let semantic_text = semantic_query_text(action).ok_or_else(|| {
                format!(
                    "action '{}' requested semantic mode but provided no semantic/text/intent input",
                    action.id
                )
            })?;
            let query_vector = embed_query(embedder, semantic_text)?;
            plan_to_semantic_query(action, query_vector)
        }
    }
}

/// Translate an action into a keyword query (`CONTAINS`).
fn plan_to_keyword_query(action: &ActionPlan) -> Result<PlannedQuery, String> {
    let (partition_field, partition_value) = partition_for(action)?;

    let mut params: Vec<QueryParam> = vec![QueryParam::new("@key", partition_value.clone())];

    // Start the WHERE clause with the partition-key equality.
    let mut predicates = vec![format!("c.{partition_field} = @key")];

    // Optional lower/upper date bounds (the `/date` field is `yyyy-mm-dd`).
    if let Some(from) = non_empty(&action.filters.from_date) {
        predicates.push("c.date >= @from".to_string());
        params.push(QueryParam::new("@from", from));
    }
    if let Some(to) = non_empty(&action.filters.to_date) {
        predicates.push("c.date <= @to".to_string());
        params.push(QueryParam::new("@to", to));
    }

    // Optional case-insensitive substring match over the record text.
    if let Some(text) = non_empty(&action.filters.text) {
        let text_field = keyword_text_field_for(&action.target);
        predicates.push(format!("CONTAINS(LOWER(c.{text_field}), @text)"));
        params.push(QueryParam::new("@text", text.to_lowercase()));
    }

    // `TOP` takes an integer literal (program-owned, so safe to interpolate);
    // every user-supplied value above is bound as a parameter instead.
    // The Coconut fields (salience, aakkk, tokenCount) are projected so the
    // query-time KB reduction can rank by salience and pack by stored tokens.
    let sql = format!(
        "SELECT TOP {top} c.id, c.entity, c.userId, c.date, c.data, c.dataVector, c.salience, c.aakkk, c.tokenCount \
         FROM c WHERE {where_clause} ORDER BY c.date DESC",
        top = action.effective_top(),
        where_clause = predicates.join(" AND "),
    );

    Ok(PlannedQuery {
        partition_value,
        sql,
        params,
    })
}

/// Translate an action into a native Cosmos semantic query (`VectorDistance`).
fn plan_to_semantic_query(
    action: &ActionPlan,
    query_vector: Vec<f32>,
) -> Result<PlannedQuery, String> {
    let (partition_field, partition_value) = partition_for(action)?;

    let mut params: Vec<QueryParam> = vec![QueryParam::new("@key", partition_value.clone())];

    let mut predicates = vec![format!("c.{partition_field} = @key")];
    if let Some(from) = non_empty(&action.filters.from_date) {
        predicates.push("c.date >= @from".to_string());
        params.push(QueryParam::new("@from", from));
    }
    if let Some(to) = non_empty(&action.filters.to_date) {
        predicates.push("c.date <= @to".to_string());
        params.push(QueryParam::new("@to", to));
    }
    predicates.push("IS_DEFINED(c.dataVector)".to_string());
    params.push(query_vector_param(query_vector));

    let distance_expr =
        format!("VectorDistance(c.dataVector, @queryVector, false, {VECTOR_DISTANCE_OPTIONS})");

    // Project the Coconut fields (salience, aakkk, tokenCount) alongside the
    // Cosmos-computed similarityScore so the KB reduction can rank by the full
    // salience × similarity formula and pack by stored token counts.
    let sql = format!(
        "SELECT TOP {top} c.id, c.entity, c.userId, c.date, c.data, c.dataVector, c.salience, c.aakkk, c.tokenCount, {distance_expr} AS similarityScore \
         FROM c WHERE {where_clause} ORDER BY {distance_expr}",
        top = action.effective_top(),
        where_clause = predicates.join(" AND "),
    );

    Ok(PlannedQuery {
        partition_value,
        sql,
        params,
    })
}

/// Build a query parameter for a vector embedding.
fn query_vector_param(vector: Vec<f32>) -> QueryParam {
    let values = vector
        .into_iter()
        .map(|value| serde_json::Value::from(f64::from(value)))
        .collect::<Vec<_>>();
    QueryParam::new("@queryVector", serde_json::Value::Array(values))
}

/// Embed query text using the configured embedder.
fn embed_query(embedder: Option<&EmbeddingClient>, text: &str) -> Result<Vec<f32>, String> {
    let embedder = embedder.ok_or_else(|| {
        "semantic mode needs embeddings, but EMBEDDING_DEPLOYMENT/credentials are not configured"
            .to_string()
    })?;
    embedder.embed(text).map_err(|err| err.to_string())
}

/// Return whether the `GAIA_FORCE_SEMANTIC` override is enabled.
///
/// When set to a truthy value, callers rewrite every supported retrieval to
/// semantic (vector) mode before executing, overriding the model's authored
/// keyword query and `filters.mode`. This is the on switch behind
/// `infra/TestDataRetrieval.ps1`'s semantic run; default (unset) preserves the
/// model-authored behaviour.
pub fn force_semantic() -> bool {
    force_semantic_from(crate::llm::value_from_env("GAIA_FORCE_SEMANTIC").as_deref())
}

/// Pure parse of the `GAIA_FORCE_SEMANTIC` value: `1|true|yes|on` (any case,
/// surrounding whitespace ignored) means enabled; everything else is off.
fn force_semantic_from(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Rewrite every supported action in `plan` to semantic (vector) retrieval.
///
/// This mutates the plan **in place** so the change is visible both in the saved
/// `actions.json` artifact and in the query that actually runs: it drops any
/// model-authored keyword `query` and sets `filters.mode = "semantic"`. Targets
/// that cannot be searched semantically (e.g. `GaiaConnections`, which has no
/// embedding) are left untouched so they keep their normal keyword path.
pub fn force_semantic_on(plan: &mut ActionsFile) {
    for action in &mut plan.actions {
        if target_supports_semantic(&action.target) {
            // Drop the authored keyword SQL so the structured semantic builder runs.
            action.query = None;
            action.filters.mode = Some("semantic".to_string());
        }
    }
}

/// The Gaia containers that always hold Gaia's core knowledge about a user:
/// her knowledge-base facts about them ([`GaiaKB`]-equivalent), her diary
/// reflections, and the connection ledger that tracks the relationship.
///
/// These are the three sources that answer "who is this person?" — so Gaia must
/// read them on *every* turn, regardless of what LLM Call 1 plans.
pub const CORE_USER_CONTAINERS: [&str; 3] = ["GaiaKB", "GaiaDiary", "GaiaConnections"];

/// Default top-N for a forced core-user query.
///
/// Small on purpose: we want the most recent few records per source to remind
/// Gaia of the user's identity, not flood the Response Data Context.
const CORE_USER_QUERY_TOP: usize = 3;

/// Guarantee that this turn retrieves Gaia's core knowledge about the user.
///
/// Gaia must never "forget who you are". LLM Call 1 is free to plan whatever
/// extra retrieval a question needs, but it sometimes omits the queries that
/// fetch the user's identity context — or scopes them to the wrong entity. To
/// make every turn robust, this appends a guaranteed query for each core
/// container in [`CORE_USER_CONTAINERS`] that the plan does not already cover
/// for `user_id`, pinned to that user (`entity == user_id`, the migration's
/// partition key). Containers the model already queries for this user are left
/// untouched so it can still refine them.
///
/// Each forced query uses keyword mode with no text filter, which the executor
/// turns into "the most recent records in the user's partition" — so it works
/// even when embeddings are not configured. When `GAIA_FORCE_SEMANTIC` is on,
/// call this *before* [`force_semantic_on`] so the override upgrades the
/// embeddable core queries (KB, diary) too.
pub fn ensure_core_user_queries(plan: &mut ActionsFile, user_id: &str) {
    for container in CORE_USER_CONTAINERS {
        // Skip a container the model already queries for this exact user, so we
        // never duplicate a retrieval it deliberately scoped.
        let already_planned = plan.actions.iter().any(|action| {
            action.target.eq_ignore_ascii_case(container)
                && action.entity.as_deref().map(str::trim) == Some(user_id)
        });
        if already_planned {
            continue;
        }

        plan.actions.push(ActionPlan {
            // A stable, namespaced id that cannot collide with the model's q1..qN.
            id: format!("core-{}", container.to_ascii_lowercase()),
            kind: "query".to_string(),
            target: container.to_string(),
            user_id: Some(user_id.to_string()),
            entity: Some(user_id.to_string()),
            intent: format!("Core {container} context about the user"),
            top: CORE_USER_QUERY_TOP,
            query: None,
            filters: ActionFilters {
                // Keyword mode + no text filter => the most recent records in the
                // user's partition, which needs no embedding client to run.
                mode: Some("keyword".to_string()),
                ..ActionFilters::default()
            },
        });
    }
}

/// Supported retrieval modes for one action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetrievalMode {
    Keyword,
    Semantic,
}

/// Decide retrieval mode from the action filter contract.
fn mode_for(action: &ActionPlan) -> Result<RetrievalMode, String> {
    match non_empty(&action.filters.mode)
        .unwrap_or("auto")
        .to_ascii_lowercase()
        .as_str()
    {
        "keyword" => Ok(RetrievalMode::Keyword),
        "semantic" => Ok(RetrievalMode::Semantic),
        "auto" => {
            if non_empty(&action.filters.semantic).is_some() {
                Ok(RetrievalMode::Semantic)
            } else {
                Ok(RetrievalMode::Keyword)
            }
        }
        other => Err(format!(
            "unsupported filters.mode '{other}' (expected keyword|semantic|auto)"
        )),
    }
}

/// Resolve the semantic query text in descending preference order.
fn semantic_query_text(action: &ActionPlan) -> Option<&str> {
    non_empty(&action.filters.semantic)
        .or_else(|| non_empty(&action.filters.text))
        .or_else(|| {
            let intent = action.intent.trim();
            if intent.is_empty() {
                None
            } else {
                Some(intent)
            }
        })
}

/// Return whether a target container supports semantic retrieval.
fn target_supports_semantic(target: &str) -> bool {
    !target.eq_ignore_ascii_case("GaiaConnections")
}

/// Return the keyword-search field for a container.
fn keyword_text_field_for(target: &str) -> &'static str {
    if target.eq_ignore_ascii_case("GaiaConnections") {
        "notes"
    } else {
        "data"
    }
}

/// Decide the partition field + value for an action, or explain why it can't.
fn partition_for(action: &ActionPlan) -> Result<(&'static str, String), String> {
    let user_id = non_empty(&action.user_id).ok_or_else(|| {
        format!(
            "action '{}' targets {} but has no user_id",
            action.id, action.target
        )
    })?;

    let entity = non_empty(&action.entity).ok_or_else(|| {
        format!(
            "action '{}' targets {} but has no entity",
            action.id, action.target
        )
    })?;

    if user_id != entity {
        return Err(format!(
            "action '{}' has mismatched user_id '{}' and entity '{}'",
            action.id, user_id, entity
        ));
    }

    // Every container partitions on /entity (and user_id == entity is enforced
    // above), so each query is pinned to the user's own partition.
    Ok(("entity", entity.to_string()))
}

/// Return the trimmed string slice if a value is present and non-blank.
fn non_empty(value: &Option<String>) -> Option<&str> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|trimmed| !trimmed.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ActionFilters;
    use crate::actions::SessionContext;

    /// Build a minimal valid query action for tests.
    fn action(target: &str) -> ActionPlan {
        ActionPlan {
            id: "q1".to_string(),
            kind: "query".to_string(),
            target: target.to_string(),
            user_id: None,
            entity: None,
            intent: "find things".to_string(),
            top: 0,
            query: None,
            filters: ActionFilters::default(),
        }
    }

    #[test]
    fn entity_action_partitions_on_entity_with_default_top() {
        let mut act = action("GaiaKB");
        act.user_id = Some("user-1".to_string());
        act.entity = Some("user-1".to_string());

        let planned = plan_to_query(&act, None).unwrap();

        assert_eq!(planned.partition_value, "user-1");
        assert!(planned.sql.contains("SELECT TOP 3 "));
        assert!(planned.sql.contains("c.entity = @key"));
        assert!(planned.sql.contains("ORDER BY c.date DESC"));
        assert_eq!(planned.params, vec![QueryParam::new("@key", "user-1")]);
    }

    #[test]
    fn gaia_action_partitions_on_entity_and_applies_filters() {
        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        act.top = 5;
        act.filters = ActionFilters {
            from_date: Some("2026-06-01".to_string()),
            to_date: Some("2026-06-16".to_string()),
            text: Some("Borrow Checker".to_string()),
            semantic: None,
            mode: Some("keyword".to_string()),
        };

        let planned = plan_to_query(&act, None).unwrap();

        assert_eq!(planned.partition_value, "rust");
        assert!(planned.sql.contains("SELECT TOP 5 "));
        assert!(planned.sql.contains("c.entity = @key"));
        assert!(planned.sql.contains("c.date >= @from"));
        assert!(planned.sql.contains("c.date <= @to"));
        assert!(planned.sql.contains("CONTAINS(LOWER(c.data), @text)"));
        // Free text is lowercased before binding.
        assert!(planned
            .params
            .contains(&QueryParam::new("@text", "borrow checker")));
        assert!(planned
            .params
            .contains(&QueryParam::new("@from", "2026-06-01")));
    }

    #[test]
    fn action_without_user_id_is_an_error() {
        let mut act = action("GaiaKB");
        act.entity = Some("user-1".to_string());
        let err = plan_to_query(&act, None).unwrap_err();
        assert!(err.contains("no user_id"));
    }

    #[test]
    fn gaia_action_without_entity_is_an_error() {
        let mut act = action("GaiaDiary");
        act.user_id = Some("threadkeeper".to_string());
        let err = plan_to_query(&act, None).unwrap_err();
        assert!(err.contains("no entity"));
    }

    #[test]
    fn mismatched_user_id_and_entity_is_an_error() {
        let mut act = action("GaiaDiary");
        act.user_id = Some("threadkeeper".to_string());
        act.entity = Some("jonty".to_string());
        let err = plan_to_query(&act, None).unwrap_err();
        assert!(err.contains("mismatched user_id"));
    }

    #[test]
    fn blank_filters_are_dropped() {
        let mut act = action("GaiaKB");
        act.user_id = Some("user-1".to_string());
        act.entity = Some("user-1".to_string());
        act.filters.text = Some("   ".to_string());

        let planned = plan_to_query(&act, None).unwrap();

        // A whitespace-only text filter must not add a CONTAINS predicate.
        assert!(!planned.sql.contains("CONTAINS"));
        assert_eq!(planned.params, vec![QueryParam::new("@key", "user-1")]);
    }

    #[test]
    fn plan_for_prefers_the_authored_query_and_binds_the_partition() {
        let mut act = action("GaiaDiary");
        act.user_id = Some("threadkeeper".to_string());
        act.entity = Some("threadkeeper".to_string());
        act.query = Some(
            "SELECT TOP 3 c.id, c.entity, c.date, c.data FROM c \
             WHERE c.entity = @pk AND CONTAINS(LOWER(c.data), 'tree') ORDER BY c.date DESC"
                .to_string(),
        );

        let planned = plan_for(&act, None).unwrap();

        // The model's exact SQL is used verbatim.
        assert!(planned.sql.contains("CONTAINS(LOWER(c.data), 'tree')"));
        // The query is still pinned to a single partition: @pk is bound to the entity.
        assert_eq!(planned.partition_value, "threadkeeper");
        assert_eq!(planned.params, vec![QueryParam::new("@pk", "threadkeeper")]);
    }

    #[test]
    fn plan_for_does_not_bind_pk_when_the_query_inlines_the_partition() {
        let mut act = action("GaiaDiary");
        act.user_id = Some("threadkeeper".to_string());
        act.entity = Some("threadkeeper".to_string());
        // No `@pk` placeholder: the model inlined the partition value itself.
        act.query = Some("SELECT c.id, c.data FROM c WHERE c.entity = 'threadkeeper'".to_string());

        let planned = plan_for(&act, None).unwrap();

        // Nothing to bind, but the partition value is still resolved for the header.
        assert!(planned.params.is_empty());
        assert_eq!(planned.partition_value, "threadkeeper");
    }

    #[test]
    fn plan_for_falls_back_to_the_built_query_when_none_is_authored() {
        let mut act = action("GaiaKB");
        act.user_id = Some("user-1".to_string());
        act.entity = Some("user-1".to_string());
        // No authored query -> the structured-field builder is used.
        let planned = plan_for(&act, None).unwrap();
        assert!(planned.sql.contains("SELECT TOP 3 "));
        assert!(planned.sql.contains("c.entity = @key"));
    }

    #[test]
    fn an_authored_non_select_query_is_rejected() {
        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        // Even though Cosmos cannot run this, we reject it early and clearly.
        act.query = Some("DELETE FROM c WHERE c.entity = 'rust'".to_string());
        let err = plan_for(&act, None).unwrap_err();
        assert!(err.contains("SELECT"));
    }

    #[test]
    fn an_authored_query_with_multiple_statements_is_rejected() {
        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        act.query = Some("SELECT * FROM c; SELECT * FROM c".to_string());
        let err = plan_for(&act, None).unwrap_err();
        assert!(err.contains("single statement"));
    }

    #[test]
    fn semantic_mode_builds_native_vectordistance_query() {
        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        act.top = 4;
        act.filters.mode = Some("semantic".to_string());
        act.filters.semantic = Some("borrow checker ownership".to_string());

        let planned = plan_to_semantic_query(&act, vec![0.1, 0.2, 0.3]).unwrap();

        assert!(planned.sql.contains("SELECT TOP 4"));
        assert!(planned
            .sql
            .contains("VectorDistance(c.dataVector, @queryVector"));
        assert!(planned.sql.contains("searchListSizeMultiplier:10"));
        assert!(planned
            .sql
            .contains("ORDER BY VectorDistance(c.dataVector, @queryVector"));
        assert!(planned.sql.contains("IS_DEFINED(c.dataVector)"));
        assert_eq!(planned.partition_value, "rust");
        assert!(planned
            .params
            .iter()
            .any(|param| param.name.eq("@queryVector")));
    }

    #[test]
    fn semantic_mode_without_embedder_is_a_clear_error() {
        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        act.filters.mode = Some("semantic".to_string());
        act.filters.semantic = Some("lifetimes".to_string());

        let err = plan_to_query(&act, None).unwrap_err();
        assert!(err.contains("semantic mode needs embeddings"));
    }

    #[test]
    fn force_semantic_from_recognises_truthy_values() {
        for value in ["1", "true", "TRUE", "Yes", " on "] {
            assert!(force_semantic_from(Some(value)), "expected '{value}' on");
        }
        for value in [None, Some(""), Some("0"), Some("false"), Some("off")] {
            assert!(!force_semantic_from(value), "expected {value:?} off");
        }
    }

    #[test]
    fn forced_semantic_overrides_an_authored_keyword_query() {
        let mut act = action("GaiaDataLake");
        act.user_id = Some("threadkeeper".to_string());
        act.entity = Some("threadkeeper".to_string());
        // The model authored a keyword CONTAINS query; the force must drop it.
        act.query = Some(
            "SELECT TOP 12 c.id, c.data FROM c WHERE c.entity = @pk \
             AND CONTAINS(LOWER(c.data), 'music') ORDER BY c.date DESC"
                .to_string(),
        );
        act.filters.text = Some("music books outdoors".to_string());
        act.filters.mode = Some("keyword".to_string());

        let mut plan = ActionsFile {
            version: "1.0".to_string(),
            session: SessionContext {
                user_id: "threadkeeper".to_string(),
                requested_at: "2026-06-21T00:00:00Z".to_string(),
            },
            actions: vec![act],
        };
        force_semantic_on(&mut plan);

        // The authored keyword query is gone and the mode flips to semantic, so
        // both the saved artifact and the executed query are semantic.
        assert_eq!(plan.actions[0].query, None);
        assert_eq!(plan.actions[0].filters.mode.as_deref(), Some("semantic"));
    }

    #[test]
    fn forced_semantic_leaves_unsupported_targets_unchanged() {
        let mut act = action("GaiaConnections");
        act.user_id = Some("threadkeeper".to_string());
        act.entity = Some("threadkeeper".to_string());
        act.filters.text = Some("kindness".to_string());
        act.filters.mode = Some("keyword".to_string());

        let mut plan = ActionsFile {
            version: "1.0".to_string(),
            session: SessionContext {
                user_id: "threadkeeper".to_string(),
                requested_at: "2026-06-21T00:00:00Z".to_string(),
            },
            actions: vec![act],
        };
        force_semantic_on(&mut plan);

        // GaiaConnections has no embedding, so it stays on the keyword path.
        assert_eq!(plan.actions[0].filters.mode.as_deref(), Some("keyword"));
    }

    #[test]
    fn ensure_core_user_queries_appends_all_three_core_containers() {
        // An empty plan must gain a guaranteed query for every core container,
        // each pinned to the user (entity == user_id == the partition key).
        let mut plan = ActionsFile::default();
        ensure_core_user_queries(&mut plan, "threadkeeper");

        assert_eq!(plan.actions.len(), 3);
        for container in CORE_USER_CONTAINERS {
            let action = plan
                .actions
                .iter()
                .find(|a| a.target == container)
                .unwrap_or_else(|| panic!("missing forced query for {container}"));
            assert_eq!(action.kind, "query");
            assert_eq!(action.user_id.as_deref(), Some("threadkeeper"));
            assert_eq!(action.entity.as_deref(), Some("threadkeeper"));
            // Keyword mode + no text filter keeps the query runnable without an
            // embedding client.
            assert_eq!(action.filters.mode.as_deref(), Some("keyword"));
            assert!(action.filters.text.is_none());
            // The forced action passes the strict user-isolation contract.
            assert!(action.validate().is_ok());
        }
    }

    #[test]
    fn ensure_core_user_queries_does_not_duplicate_an_existing_query() {
        // The model already queries GaiaKB for this user, so only the two other
        // core containers are appended; the model's action is left untouched.
        let mut existing = action("GaiaKB");
        existing.id = "q5".to_string();
        existing.user_id = Some("alice".to_string());
        existing.entity = Some("alice".to_string());
        existing.filters.semantic = Some("favourite books".to_string());

        let mut plan = ActionsFile {
            version: "1.0".to_string(),
            session: SessionContext {
                user_id: "alice".to_string(),
                requested_at: "2026-06-21T00:00:00Z".to_string(),
            },
            actions: vec![existing.clone()],
        };
        ensure_core_user_queries(&mut plan, "alice");

        assert_eq!(plan.actions.len(), 3);
        // The model's GaiaKB action is preserved verbatim (not overwritten).
        assert_eq!(plan.actions[0], existing);
        // The other two core containers were appended.
        assert!(plan.actions.iter().any(|a| a.target == "GaiaDiary"));
        assert!(plan.actions.iter().any(|a| a.target == "GaiaConnections"));
        assert_eq!(
            plan.actions.iter().filter(|a| a.target == "GaiaKB").count(),
            1
        );
    }

    #[test]
    fn ensure_core_user_queries_adds_a_query_for_a_different_user() {
        // A core query the model scoped to a *different* user does not satisfy
        // the current user's need, so a fresh query is still appended.
        let mut other = action("GaiaDiary");
        other.user_id = Some("bob".to_string());
        other.entity = Some("bob".to_string());

        let mut plan = ActionsFile {
            version: "1.0".to_string(),
            session: SessionContext::default(),
            actions: vec![other],
        };
        ensure_core_user_queries(&mut plan, "alice");

        // bob's GaiaDiary plus alice's three forced core queries.
        assert_eq!(plan.actions.len(), 4);
        assert!(plan
            .actions
            .iter()
            .any(|a| a.target == "GaiaDiary" && a.entity.as_deref() == Some("alice")));
    }

    #[test]
    fn gaia_connections_keyword_text_filter_targets_notes() {
        let mut act = action("GaiaConnections");
        act.user_id = Some("threadkeeper".to_string());
        act.entity = Some("threadkeeper".to_string());
        act.filters.mode = Some("keyword".to_string());
        act.filters.text = Some("trust".to_string());

        let planned = plan_to_query(&act, None).unwrap();
        assert!(planned.sql.contains("CONTAINS(LOWER(c.notes), @text)"));
    }

    #[test]
    fn invalid_mode_is_rejected() {
        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        act.filters.mode = Some("hybrid".to_string());

        let err = plan_to_query(&act, None).unwrap_err();
        assert!(err.contains("keyword|semantic|auto"));
    }

    /// Wrap a single action in a one-action plan for the end-to-end run tests.
    fn plan_with(act: ActionPlan) -> ActionsFile {
        ActionsFile {
            version: "1.0".to_string(),
            session: SessionContext {
                user_id: "rust".to_string(),
                requested_at: "2026-06-21T00:00:00Z".to_string(),
            },
            actions: vec![act],
        }
    }

    #[test]
    fn run_executes_a_keyword_action_against_the_client() {
        // A keyword action plans without an embedder, runs one query, and parses
        // the Documents envelope the mock returns.
        let body = r#"{"Documents":[
            {"id":"GaiaKB|rust|2026-05-10","entity":"rust","date":"2026-05-10","data":"ownership"}
        ]}"#;
        let (endpoint, handle) = crate::test_http::spawn_mock_http("200 OK", body);
        let client = CosmosClient::new(endpoint, "gaia", "tok");

        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        let outcomes = Executor::new(&client).run(&plan_with(act));

        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].id, "q1");
        let records = outcomes[0].result.as_ref().expect("the query succeeds");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].data, "ownership");
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn run_captures_a_cosmos_error_per_action_without_aborting() {
        // A non-success status from Cosmos becomes the action's own error; the
        // run still returns one outcome.
        let (endpoint, handle) =
            crate::test_http::spawn_mock_http("403 Forbidden", r#"{"message":"denied"}"#);
        let client = CosmosClient::new(endpoint, "gaia", "tok");

        let mut act = action("GaiaKB");
        act.user_id = Some("rust".to_string());
        act.entity = Some("rust".to_string());
        let outcomes = Executor::new(&client).run(&plan_with(act));

        assert_eq!(outcomes.len(), 1);
        let err = outcomes[0]
            .result
            .as_ref()
            .expect_err("a 403 fails the action");
        assert!(err.contains("403"), "got: {err}");
        handle.join().expect("mock server thread joins");
    }

    #[test]
    fn run_reports_a_validation_error_without_calling_the_client() {
        // A GaiaKB action with no entity fails validation before any HTTP call,
        // so no mock server is needed.
        let client = CosmosClient::new("http://127.0.0.1:0/", "gaia", "tok");
        let act = action("GaiaKB"); // no user_id / entity set
        let outcomes = Executor::new(&client).run(&plan_with(act));

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].result.is_err());
    }
}
