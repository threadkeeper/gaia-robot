//! Data contract for the `actions.json` file produced by the LLM.
//!
//! The runner uses this structure to decide which Cosmos queries to execute and
//! how to thread the results back into an in-memory response model. The schema
//! intentionally keeps each action explicit: the target container, the user
//! partition, the natural-language intent, and the top-N limit.

use serde::{Deserialize, Serialize};

/// The top-level JSON document emitted by the first LLM pass.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ActionsFile {
    /// Schema version for the contract.
    pub version: String,
    /// Session-scoped context for the request.
    pub session: SessionContext,
    /// One or more query actions to execute.
    pub actions: Vec<ActionPlan>,
}

/// Session metadata that accompanies the action plan.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SessionContext {
    /// The caller user identifier; used to guarantee user isolation.
    pub user_id: String,
    /// The time the request was generated, if available.
    pub requested_at: String,
}

/// A single action to execute against the in-memory Cosmos abstraction.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ActionPlan {
    /// Stable identifier for the action, e.g. `q1`.
    pub id: String,
    /// The action family; today this should be `query`.
    pub kind: String,
    /// The target container, such as `UsersKB` or `GaiaKB`.
    pub target: String,
    /// The user partition for `Users*` containers.
    pub user_id: Option<String>,
    /// The entity or subject to search for, when relevant.
    pub entity: Option<String>,
    /// The natural-language intent to translate into a query.
    pub intent: String,
    /// Maximum number of results to return. Defaults to `3`.
    pub top: usize,
    /// Optional filters to refine retrieval.
    pub filters: ActionFilters,
}

/// Optional filters attached to a query action.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ActionFilters {
    /// Lower bound of the date range, if provided.
    pub from_date: Option<String>,
    /// Upper bound of the date range, if provided.
    pub to_date: Option<String>,
    /// Free-text keyword filter, if provided.
    pub text: Option<String>,
    /// Semantic hint for similarity or vector-oriented retrieval.
    pub semantic: Option<String>,
}

#[allow(dead_code)]
impl ActionPlan {
    /// Return a safe top-N value for execution.
    pub fn effective_top(&self) -> usize {
        if self.top == 0 {
            3
        } else {
            self.top
        }
    }

    /// Ensure the action is runnable and user-isolated where required.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.kind != "query" {
            return Err("kind must be 'query'");
        }

        if self.target.is_empty() {
            return Err("target must be set");
        }

        if self.effective_top() == 0 {
            return Err("top must be at least 1");
        }

        if matches!(self.target.as_str(), "UsersKB" | "UsersDL")
            && self.user_id.as_deref().is_none()
        {
            return Err("Users* actions require user_id");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_top_is_three_when_not_supplied() {
        let action = ActionPlan {
            id: "q1".to_string(),
            kind: "query".to_string(),
            target: "UsersKB".to_string(),
            user_id: Some("user-1".to_string()),
            entity: None,
            intent: "Recent notes".to_string(),
            top: 0,
            filters: ActionFilters::default(),
        };

        assert_eq!(action.effective_top(), 3);
        assert!(action.validate().is_ok());
    }

    #[test]
    fn users_actions_require_a_user_id() {
        let action = ActionPlan {
            id: "q2".to_string(),
            kind: "query".to_string(),
            target: "UsersDL".to_string(),
            user_id: None,
            entity: None,
            intent: "Recent activity".to_string(),
            top: 3,
            filters: ActionFilters::default(),
        };

        assert!(action.validate().is_err());
    }

    #[test]
    fn json_round_trips_with_the_expected_contract() {
        let raw = r#"{
            "version": "1.0",
            "session": {"user_id": "user-123", "requested_at": "2026-06-16T12:00:00Z"},
            "actions": [
                {
                    "id": "q1",
                    "kind": "query",
                    "target": "UsersKB",
                    "user_id": "user-123",
                    "entity": "notes",
                    "intent": "Recent notes for this user",
                    "top": 3,
                    "filters": {"from_date": "2026-06-01", "to_date": "2026-06-16"}
                }
            ]
        }"#;

        let parsed: ActionsFile = serde_json::from_str(raw).unwrap();

        assert_eq!(parsed.version, "1.0");
        assert_eq!(parsed.session.user_id, "user-123");
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.actions[0].effective_top(), 3);
        assert!(parsed.actions[0].validate().is_ok());
    }
}
