// This contract module is now wired into `main`'s loop: LLM Call 1's `Web`
// action records each search here via `record`. Some convenience accessors
// (`recent`, `is_empty`, ...) are still unused, so keep the module-level
// allow to avoid dead_code churn until the executor consumes the full log.
#![allow(dead_code)]

//! The **Gaia Search History**: a simple, append-only log of Gaia's web
//! searches and the results they returned.
//!
//! This store exists purely for **logging / auditing**. When Gaia runs a web
//! search during the GET / pull pass, we append the query and the results here
//! so the searches are visible and replayable after the fact.
//!
//! Deliberately *not* a retrieval store: entries carry **no embedding / vector**
//! for now, because nothing does semantic search over this history yet. It is a
//! plain chronological log, oldest first, living in Gaia's own space (the
//! `GaiaSearchHistory` store in the architecture diagram).

use serde::{Deserialize, Serialize};

/// A single result returned by a web search.
///
/// Kept intentionally small — just enough to record what Gaia saw, not to power
/// any ranking or retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The result's title / headline.
    pub title: String,
    /// The result's URL.
    pub url: String,
    /// A short snippet / summary of the result.
    pub snippet: String,
}

impl SearchResult {
    /// Build a single search result from its parts.
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
    ) -> Self {
        SearchResult {
            title: title.into(),
            url: url.into(),
            snippet: snippet.into(),
        }
    }
}

/// One logged web search: the query Gaia ran and the results it returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHistoryEntry {
    /// When the search was performed. Expected to be lexicographically
    /// sortable (e.g. ISO-8601 / RFC-3339) so the log stays in time order.
    pub timestamp: String,
    /// The query Gaia searched the web for.
    pub query: String,
    /// The results returned, in the order the search engine ranked them.
    /// Logged only — these are never embedded or indexed for retrieval.
    pub results: Vec<SearchResult>,
}

/// An append-only log of Gaia's web searches, oldest first.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHistory {
    // Entries in the order they were recorded (time order). A plain `Vec` is
    // all we need: this is an append-only log, never keyed or de-duplicated.
    entries: Vec<SearchHistoryEntry>,
}

impl SearchHistory {
    /// Append a search and its results to the log, returning the stored entry.
    pub fn record(
        &mut self,
        timestamp: impl Into<String>,
        query: impl Into<String>,
        results: Vec<SearchResult>,
    ) -> &SearchHistoryEntry {
        self.entries.push(SearchHistoryEntry {
            timestamp: timestamp.into(),
            query: query.into(),
            results,
        });
        // We just pushed, so the log is non-empty and `len() - 1` is in range.
        let last_index = self.entries.len() - 1;
        &self.entries[last_index]
    }

    /// All logged searches, oldest first.
    pub fn entries(&self) -> &[SearchHistoryEntry] {
        &self.entries
    }

    /// The most recent `n` searches (fewer if the log is shorter), oldest
    /// first within the returned slice.
    pub fn recent(&self, n: usize) -> &[SearchHistoryEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Number of searches logged so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been logged yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A couple of small results used across the tests.
    fn sample_results() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                "Current time in Johannesburg",
                "https://example.com/time/jhb",
                "It is 17:21 in Johannesburg (SAST, UTC+2).",
            ),
            SearchResult::new(
                "South Africa time zone",
                "https://example.com/tz/za",
                "South Africa uses SAST, two hours ahead of UTC.",
            ),
        ]
    }

    #[test]
    fn a_new_log_is_empty() {
        let history = SearchHistory::default();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.entries().is_empty());
    }

    #[test]
    fn record_appends_and_returns_the_stored_entry() {
        let mut history = SearchHistory::default();
        let entry = history.record(
            "2026-05-03T15:21:54Z",
            "current time in Johannesburg",
            sample_results(),
        );

        assert_eq!(entry.query, "current time in Johannesburg");
        assert_eq!(entry.timestamp, "2026-05-03T15:21:54Z");
        assert_eq!(entry.results.len(), 2);
        assert_eq!(history.len(), 1);
        assert!(!history.is_empty());
    }

    #[test]
    fn entries_are_kept_in_the_order_they_were_recorded() {
        let mut history = SearchHistory::default();
        history.record("2026-05-03T10:00:00Z", "first", Vec::new());
        history.record("2026-05-03T11:00:00Z", "second", Vec::new());
        history.record("2026-05-03T12:00:00Z", "third", Vec::new());

        let queries: Vec<&str> = history
            .entries()
            .iter()
            .map(|entry| entry.query.as_str())
            .collect();
        assert_eq!(queries, ["first", "second", "third"]);
    }

    #[test]
    fn recent_returns_only_the_last_n_searches() {
        let mut history = SearchHistory::default();
        history.record("2026-05-03T10:00:00Z", "first", Vec::new());
        history.record("2026-05-03T11:00:00Z", "second", Vec::new());
        history.record("2026-05-03T12:00:00Z", "third", Vec::new());

        let recent: Vec<&str> = history
            .recent(2)
            .iter()
            .map(|entry| entry.query.as_str())
            .collect();
        assert_eq!(recent, ["second", "third"]);

        // Asking for more than we have just returns everything.
        assert_eq!(history.recent(10).len(), 3);
    }

    #[test]
    fn an_entry_round_trips_through_json() {
        let mut history = SearchHistory::default();
        let original = history
            .record(
                "2026-05-03T15:21:54Z",
                "current time in Johannesburg",
                sample_results(),
            )
            .clone();

        let json = serde_json::to_string(&original).expect("entry should serialize");
        let restored: SearchHistoryEntry =
            serde_json::from_str(&json).expect("entry should deserialize");

        assert_eq!(restored, original);
    }
}
