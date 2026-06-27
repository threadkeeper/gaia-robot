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
    /// The user id that scopes this query.
    ///
    /// Strict contract: every query action must set this and it must match
    /// [`ActionPlan::entity`].
    pub user_id: Option<String>,
    /// The entity id this query is scoped to.
    ///
    /// Strict contract: every query action must set this and it must match
    /// [`ActionPlan::user_id`].
    pub entity: Option<String>,
    /// The natural-language intent to translate into a query.
    pub intent: String,
    /// Maximum number of results to return. Defaults to `3`.
    pub top: usize,
    /// The exact Cosmos DB SQL the LLM authored for this action.
    ///
    /// LLM Call 1 understands the full container schema (see
    /// [`crate::prompt`]) and emits the precise read-only query to run for this
    /// retrieval — a keyword search (`CONTAINS`) or, once embeddings exist, a
    /// vector search (`VectorDistance`). The runner still scopes execution to a
    /// single logical partition via the partition-key header, so this query can
    /// only ever read within that partition. `None` (or blank) means "no query
    /// authored"; the executor then falls back to building one from the
    /// structured fields below. `Web` actions carry no query (handled by the
    /// web-search applet, not Cosmos).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Optional filters to refine retrieval.
    pub filters: ActionFilters,
}

/// Extract the `actions.json` document from LLM Call 1's raw reply.
///
/// LLM Call 1 emits a JSON array of four documents; the first is `actions.json`.
/// This locates the array by its outer brackets (tolerating any code fences or
/// stray prose around it), parses it, and deserializes element 0 into an
/// [`ActionsFile`]. Returns `None` when anything about that shape is off, so a
/// caller can degrade gracefully rather than fail the turn.
///
/// # Examples
///
/// ```
/// # // (Doc-tested indirectly via the module's unit tests, since this is a
/// # // binary crate and the function is not exported as a library API.)
/// ```
pub fn parse_call1_actions(reply: &str) -> Option<ActionsFile> {
    // Reuse the shared array extractor, then take element 0 (actions.json).
    let documents = extract_call1_array(reply)?;
    let first = documents.first()?;
    serde_json::from_value(first.clone()).ok()
}

/// Extract LLM Call 1's full four-document array from its raw reply.
///
/// LLM Call 1 emits a single JSON array of four documents (in order:
/// `actions.json`, `analysis.json`, `facts.json`, `newContext.json`). This finds
/// that array by its outer brackets — tolerating any code fences or stray prose
/// around it — parses it, and returns its elements. When the strict parse fails,
/// it makes one best-effort [`crate::json_repair::repair_json`] attempt to fix
/// the common structural defects models emit (a dropped element brace, a
/// trailing comma, a truncated tail) before giving up. Returns `None` only when
/// no array can be recovered, so callers can degrade gracefully. Both
/// [`parse_call1_actions`] and the Response Data Context builder share this so
/// they always read the *same* array the same way.
pub fn extract_call1_array(reply: &str) -> Option<Vec<serde_json::Value>> {
    // Walk forward from the first `[` counting brackets (and respecting string
    // literals) to find the matching `]`; the model sometimes appends extra
    // prose after the array, so a first-`[`-to-last-`]` scan would be invalid.
    // When the closing bracket cannot be located (e.g. the reply was truncated)
    // we fall back to the whole tail and let the repair pass close it.
    let start = reply.find('[')?;
    let rest = &reply[start..];
    let candidate = match find_balanced_bracket(rest) {
        Some(end) => &rest[..=end],
        None => rest,
    };

    // Fast path: the reply is already well-formed.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(candidate) {
        return value.as_array().cloned();
    }

    // Fallback: language models occasionally drop a single structural character
    // at an element boundary (most often the opening `{` of an array element).
    // Repair that family of defects and try once more, so one dropped brace does
    // not throw away an otherwise-good turn. The repair is a no-op on valid JSON.
    let repaired = crate::json_repair::repair_json(candidate);
    let value: serde_json::Value = serde_json::from_str(&repaired).ok()?;
    value.as_array().cloned()
}

/// Find the index (relative to `s`) of the `]` that balances the leading `[`.
///
/// Respects JSON string literals (including escaped quotes) so that brackets
/// inside strings are not counted.
fn find_balanced_bracket(s: &str) -> Option<usize> {
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
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
    /// Retrieval mode chosen by the model for this action.
    ///
    /// Supported values are `"keyword"`, `"semantic"`, or `"auto"`.
    /// `None` is treated the same as `"auto"`.
    pub mode: Option<String>,
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

    /// Return the LLM-authored Cosmos SQL if one was provided and non-blank.
    ///
    /// Returns `None` when the model did not author a query (so the executor
    /// should build one from the structured fields instead).
    pub fn authored_query(&self) -> Option<&str> {
        self.query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
    }

    /// Ensure the action is runnable and user-isolated.
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

        let user_id = self
            .user_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or("all query actions require user_id")?;

        let entity = self
            .entity
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or("all query actions require entity")?;

        if user_id != entity {
            return Err("user_id must match entity");
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
            entity: Some("user-1".to_string()),
            intent: "Recent notes".to_string(),
            top: 0,
            query: None,
            filters: ActionFilters::default(),
        };

        assert_eq!(action.effective_top(), 3);
        assert!(action.validate().is_ok());
    }

    #[test]
    fn all_query_actions_require_a_user_id() {
        let action = ActionPlan {
            id: "q2".to_string(),
            kind: "query".to_string(),
            target: "UsersDataLake".to_string(),
            user_id: None,
            entity: Some("user-1".to_string()),
            intent: "Recent activity".to_string(),
            top: 3,
            query: None,
            filters: ActionFilters::default(),
        };

        assert!(action.validate().is_err());
    }

    #[test]
    fn all_query_actions_require_entity_matching_user_id() {
        let action = ActionPlan {
            id: "q2".to_string(),
            kind: "query".to_string(),
            target: "GaiaDataLake".to_string(),
            user_id: Some("threadkeeper".to_string()),
            entity: Some("jonty".to_string()),
            intent: "Recent activity".to_string(),
            top: 3,
            query: None,
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
                    "entity": "user-123",
                    "intent": "Recent notes for this user",
                    "top": 3,
                    "query": "SELECT TOP 3 c.id, c.userId, c.date, c.data FROM c WHERE c.userId = @pk AND CONTAINS(LOWER(c.data), 'notes') ORDER BY c.date DESC",
                    "filters": {"from_date": "2026-06-01", "to_date": "2026-06-16"}
                }
            ]
        }"#;

        let parsed: ActionsFile = serde_json::from_str(raw).unwrap();

        assert_eq!(parsed.version, "1.0");
        assert_eq!(parsed.session.user_id, "user-123");
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.actions[0].effective_top(), 3);
        // The LLM-authored query is preserved verbatim and surfaced via the helper.
        assert_eq!(
            parsed.actions[0].authored_query(),
            Some(
                "SELECT TOP 3 c.id, c.userId, c.date, c.data FROM c WHERE c.userId = @pk \
                 AND CONTAINS(LOWER(c.data), 'notes') ORDER BY c.date DESC"
            )
        );
        assert!(parsed.actions[0].validate().is_ok());
    }

    #[test]
    fn a_missing_or_blank_query_is_treated_as_absent() {
        // A document with no `query` field deserializes to None.
        let raw = r#"{
            "version": "1.0",
            "session": {"user_id": "user-123", "requested_at": "2026-06-16T12:00:00Z"},
            "actions": [
                {"id": "q1", "kind": "query", "target": "GaiaKB", "user_id": "rust", "entity": "rust",
                 "intent": "x", "top": 3, "query": "   ", "filters": {}}
            ]
        }"#;

        let parsed: ActionsFile = serde_json::from_str(raw).unwrap();
        // A whitespace-only query is reported as "no query authored".
        assert_eq!(parsed.actions[0].authored_query(), None);
    }

    #[test]
    fn parse_call1_actions_extracts_the_first_document_from_fenced_prose() {
        // Call 1 emits an array of documents; element 0 is actions.json. We wrap
        // it in a code fence and prose to prove the bracket-scan is robust.
        let reply = "Sure!\n```json\n[\n  {\"version\":\"1.0\",\
            \"session\":{\"user_id\":\"threadkeeper\",\"requested_at\":\"2026-06-16T12:00:00Z\"},\
            \"actions\":[{\"id\":\"q1\",\"kind\":\"query\",\"target\":\"GaiaKB\",\
            \"user_id\":\"rust\",\"entity\":\"rust\",\"intent\":\"x\",\"top\":3,\"filters\":{}}]},\
            {\"analysis\":true},{\"facts\":[]},{\"newContext\":\"\"}\n]\n```\nthanks";

        let parsed = parse_call1_actions(reply).expect("should parse actions.json");
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.actions[0].target, "GaiaKB");
    }

    #[test]
    fn parse_call1_actions_returns_none_for_malformed_output() {
        // No JSON array, or a first element that is not an actions document.
        assert!(parse_call1_actions("no json here").is_none());
        assert!(parse_call1_actions("[123, 456]").is_none());
    }

    #[test]
    fn parse_call1_actions_handles_trailing_metadata_after_array() {
        // The model sometimes wraps the actions array inside a larger outer
        // array with extra metadata objects and commentary. The balanced-bracket
        // parser should grab only the first `[…]` and ignore the rest.
        let reply = r#"[{"version":"1.0","session":{"user_id":"threadkeeper","requested_at":"2026-06-20T10:00:00Z"},"actions":[{"id":"q6","kind":"query","target":"GaiaDiary","user_id":"threadkeeper","entity":"threadkeeper","intent":"recall","top":3,"filters":{}}]},{"emotion":"curious"},{"fact":"x","value":"y"}],"extra commentary here"}]"#;

        let parsed = parse_call1_actions(reply).expect("should parse despite trailing garbage");
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.actions[0].target, "GaiaDiary");
    }

    #[test]
    fn parse_call1_actions_recovers_from_a_dropped_element_brace() {
        // The model occasionally drops the opening `{` of an array element. Here
        // the actions document is element 0 of the outer array but its own
        // opening brace is missing (`[ "version" …` instead of `[ { "version" …`).
        // The repair fallback should restore it so the turn is not lost.
        let reply = r#"["version":"1.0","session":{"user_id":"threadkeeper","requested_at":"2026-06-20T10:00:00Z"},"actions":[{"id":"q1","kind":"query","target":"GaiaKB","user_id":"threadkeeper","entity":"threadkeeper","intent":"x","top":3,"filters":{}}]},{"analysis":true}]"#;

        let parsed = parse_call1_actions(reply).expect("repair should rescue the dropped brace");
        assert_eq!(parsed.actions.len(), 1);
        assert_eq!(parsed.actions[0].target, "GaiaKB");
    }

    #[test]
    fn extract_call1_array_recovers_from_a_dropped_brace_between_elements() {
        // A dropped `{` before the SECOND document object (`},{` became `},`).
        let reply = r#"[{"actions":[]},"analysis":{"emotion":"warm"}}]"#;

        let docs = extract_call1_array(reply).expect("repair should restore the brace");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[1]["analysis"]["emotion"], "warm");
    }
}
