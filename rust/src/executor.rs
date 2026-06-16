#![allow(dead_code)]

//! The [`Executor`] type: turns an `actions.json` plan into Cosmos queries.
//!
//! The first LLM pass emits an [`ActionsFile`] (see [`crate::actions`]). The
//! executor walks each [`ActionPlan`], translates it into a safe, parameterised
//! Cosmos SQL query, runs it through a [`CosmosClient`], and collects the
//! retrieved [`Record`]s. This is the **read** half of wiring Cosmos into the
//! program flow; writes go straight through [`CosmosClient::upsert`].
//!
//! Query construction is a pure function ([`plan_to_query`]) so it is fully unit
//! tested without a network. User-controlled values are always passed as
//! [`QueryParam`]s (never string-interpolated) to avoid query injection; the
//! only interpolated value is the integer `TOP` count, which the program owns.

use crate::actions::{ActionPlan, ActionsFile};
use crate::cosmos::{CosmosClient, QueryParam};
use crate::storage::Record;

/// The outcome of executing one [`ActionPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct ActionOutcome {
    /// The originating action's id (e.g. `q1`).
    pub id: String,
    /// The retrieved records, or an error message describing what failed.
    pub result: Result<Vec<Record>, String>,
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
}

impl<'a> Executor<'a> {
    /// Create an executor that runs against `client`.
    pub fn new(client: &'a CosmosClient) -> Self {
        Self { client }
    }

    /// Execute every action in the plan, returning one outcome per action.
    ///
    /// One failing action never aborts the others; its error is captured in its
    /// own [`ActionOutcome`].
    pub fn run(&self, actions: &ActionsFile) -> Vec<ActionOutcome> {
        actions
            .actions
            .iter()
            .map(|action| ActionOutcome {
                id: action.id.clone(),
                result: self.run_one(action),
            })
            .collect()
    }

    /// Validate, plan, and execute a single action.
    fn run_one(&self, action: &ActionPlan) -> Result<Vec<Record>, String> {
        action.validate().map_err(|err| err.to_string())?;
        let planned = plan_to_query(action)?;
        self.client
            .query(
                &action.target,
                &planned.partition_value,
                &planned.sql,
                &planned.params,
            )
            .map_err(|err| err.to_string())
    }
}

/// Translate an [`ActionPlan`] into a parameterised, single-partition query.
///
/// The partition field follows the container family: `Users*` containers
/// partition on `userId` (taken from `action.user_id`), everything else on
/// `entity` (taken from `action.entity`). Optional date-range and free-text
/// filters are appended as bound parameters. A `semantic` hint is *not* yet
/// translated into a vector search — that needs an embedding step — so it is
/// ignored for now.
pub fn plan_to_query(action: &ActionPlan) -> Result<PlannedQuery, String> {
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
        predicates.push("CONTAINS(LOWER(c.data), @text)".to_string());
        params.push(QueryParam::new("@text", text.to_lowercase()));
    }

    // `TOP` takes an integer literal (program-owned, so safe to interpolate);
    // every user-supplied value above is bound as a parameter instead.
    let sql = format!(
        "SELECT TOP {top} c.id, c.entity, c.userId, c.date, c.data, c.dataVector \
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

/// Decide the partition field + value for an action, or explain why it can't.
fn partition_for(action: &ActionPlan) -> Result<(&'static str, String), String> {
    // `Users*` containers partition on userId; all others on entity.
    if action.target.starts_with("Users") {
        match non_empty(&action.user_id) {
            Some(user_id) => Ok(("userId", user_id.to_string())),
            None => Err(format!(
                "action '{}' targets {} but has no user_id",
                action.id, action.target
            )),
        }
    } else {
        match non_empty(&action.entity) {
            Some(entity) => Ok(("entity", entity.to_string())),
            None => Err(format!(
                "action '{}' targets {} but has no entity",
                action.id, action.target
            )),
        }
    }
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
            filters: ActionFilters::default(),
        }
    }

    #[test]
    fn users_action_partitions_on_user_id_with_default_top() {
        let mut act = action("UsersKB");
        act.user_id = Some("user-1".to_string());

        let planned = plan_to_query(&act).unwrap();

        assert_eq!(planned.partition_value, "user-1");
        assert!(planned.sql.contains("SELECT TOP 3 "));
        assert!(planned.sql.contains("c.userId = @key"));
        assert!(planned.sql.contains("ORDER BY c.date DESC"));
        assert_eq!(planned.params, vec![QueryParam::new("@key", "user-1")]);
    }

    #[test]
    fn gaia_action_partitions_on_entity_and_applies_filters() {
        let mut act = action("GaiaKB");
        act.entity = Some("rust".to_string());
        act.top = 5;
        act.filters = ActionFilters {
            from_date: Some("2026-06-01".to_string()),
            to_date: Some("2026-06-16".to_string()),
            text: Some("Borrow Checker".to_string()),
            semantic: Some("ignored for now".to_string()),
        };

        let planned = plan_to_query(&act).unwrap();

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
    fn users_action_without_user_id_is_an_error() {
        let act = action("UsersDL");
        let err = plan_to_query(&act).unwrap_err();
        assert!(err.contains("no user_id"));
    }

    #[test]
    fn gaia_action_without_entity_is_an_error() {
        let act = action("GaiaLH");
        let err = plan_to_query(&act).unwrap_err();
        assert!(err.contains("no entity"));
    }

    #[test]
    fn blank_filters_are_dropped() {
        let mut act = action("UsersKB");
        act.user_id = Some("user-1".to_string());
        act.filters.text = Some("   ".to_string());

        let planned = plan_to_query(&act).unwrap();

        // A whitespace-only text filter must not add a CONTAINS predicate.
        assert!(!planned.sql.contains("CONTAINS"));
        assert_eq!(planned.params, vec![QueryParam::new("@key", "user-1")]);
    }
}
