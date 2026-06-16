// This contract module is fully modeled but not yet wired into `main`'s loop;
// allow dead_code until the connection ledger is executed at runtime (mirrors
// the other not-yet-wired contract modules such as `storage`).
#![allow(dead_code)]

//! The **Gaia Connections** ledger: Gaia's per-user "emotional bank account".
//!
//! As part of the first LLM pass, Gaia judges whether the user's input grows or
//! weakens the relationship and assigns a signed `change` (positive for a gain
//! in connection / friendship, negative for a loss). Each judgement is appended
//! to an append-only ledger that lives in Gaia's own space. Every entry records
//! the running balance, so the ledger doubles as an auditable history of the
//! relationship with each entity (the person Gaia is connecting with).
//!
//! The ledger is keyed by `(entity_id, timestamp)` — one entry per change, in
//! time order, per entity. Timestamps are expected to be lexicographically
//! sortable (e.g. ISO-8601 / RFC-3339), which is what lets us read the current
//! balance as simply "the newest entry for that entity".

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A single posting in the Gaia Connections ledger.
///
/// One entry captures a single connection judgement: the signed `change`, the
/// `previous_balance` it was applied to, the resulting `new_balance`, and a
/// short `notes` explanation of *why* Gaia moved the balance.
///
/// Balances are `f64` because Gaia's connection scores are fractional (e.g.
/// `3.9`, `9.7`); this is why the type cannot derive `Eq` (floats are only
/// `PartialEq`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionEntry {
    /// The entity (person) this connection change is about.
    pub entity_id: String,
    /// When the change was recorded; part of the `(entity, timestamp)` key.
    pub timestamp: String,
    /// Signed change in connection points: positive = gain, negative = loss.
    pub change: f64,
    /// The balance this change was applied to (the prior `new_balance`).
    pub previous_balance: f64,
    /// The resulting balance after applying `change`.
    pub new_balance: f64,
    /// Short human-readable explanation of the change.
    pub notes: String,
}

/// An append-only ledger of [`ConnectionEntry`] postings.
///
/// The ledger maintains a separate running balance per entity. Use
/// [`ConnectionLedger::record`] to post a change (it computes the previous and
/// new balances for you) and [`ConnectionLedger::balance`] to read the current
/// balance for an entity.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConnectionLedger {
    /// Postings keyed by `(entity_id, timestamp)`. The `BTreeMap` keeps entries
    /// sorted by entity and then by time, which makes "newest entry per entity"
    /// a cheap lookup.
    entries: BTreeMap<(String, String), ConnectionEntry>,
}

impl ConnectionLedger {
    /// The current connection balance for `entity_id`, or `0.0` if none exists.
    ///
    /// Because entries are sorted by `(entity, timestamp)`, the last matching
    /// entry is the newest one, and its `new_balance` is the current balance.
    pub fn balance(&self, entity_id: &str) -> f64 {
        self.entries
            .iter()
            // `rfind` walks from the newest key, so the first match is the
            // latest timestamp for this entity.
            .rfind(|((entity, _), _)| entity.as_str() == entity_id)
            .map(|(_, entry)| entry.new_balance)
            .unwrap_or(0.0)
    }

    /// Post a connection `change` for `entity_id` at `timestamp`.
    ///
    /// The previous and new balances are derived from the entity's current
    /// balance, so callers only supply the signed delta and a note. Returns the
    /// entry that was recorded.
    pub fn record(
        &mut self,
        entity_id: impl Into<String>,
        timestamp: impl Into<String>,
        change: f64,
        notes: impl Into<String>,
    ) -> ConnectionEntry {
        let entity_id = entity_id.into();
        let timestamp = timestamp.into();

        let previous_balance = self.balance(&entity_id);
        let new_balance = previous_balance + change;

        let entry = ConnectionEntry {
            entity_id: entity_id.clone(),
            timestamp: timestamp.clone(),
            change,
            previous_balance,
            new_balance,
            notes: notes.into(),
        };

        // Append the posting; the `(entity, timestamp)` key enforces one entry
        // per entity per instant.
        self.entries.insert((entity_id, timestamp), entry.clone());
        entry
    }

    /// All postings for a single entity, oldest first.
    pub fn entries_for(&self, entity_id: &str) -> Vec<&ConnectionEntry> {
        self.entries
            .iter()
            .filter(|((entity, _), _)| entity.as_str() == entity_id)
            .map(|(_, entry)| entry)
            .collect()
    }

    /// Total number of postings across all entities.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the ledger has no postings yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_is_zero_for_an_unknown_entity() {
        let ledger = ConnectionLedger::default();
        assert_eq!(ledger.balance("nobody"), 0.0);
    }

    #[test]
    fn recording_changes_accumulates_the_running_balance() {
        let mut ledger = ConnectionLedger::default();

        // Use values that are exact in binary floating point so the running
        // total can be compared precisely.
        let first = ledger.record("alice", "2026-06-16T10:00:00Z", 0.5, "warm greeting");
        assert_eq!(first.previous_balance, 0.0);
        assert_eq!(first.new_balance, 0.5);

        let second = ledger.record("alice", "2026-06-16T11:00:00Z", 0.25, "a good chat");
        assert_eq!(second.previous_balance, 0.5);
        assert_eq!(second.new_balance, 0.75);

        assert_eq!(ledger.balance("alice"), 0.75);
    }

    #[test]
    fn balances_are_tracked_per_entity() {
        let mut ledger = ConnectionLedger::default();
        ledger.record("alice", "2026-06-16T10:00:00Z", 5.0, "");
        ledger.record("bob", "2026-06-16T10:00:00Z", -1.5, "");

        assert_eq!(ledger.balance("alice"), 5.0);
        assert_eq!(ledger.balance("bob"), -1.5);
    }

    #[test]
    fn entries_for_returns_an_entitys_history_in_time_order() {
        let mut ledger = ConnectionLedger::default();
        // Insert out of order to prove the ledger sorts by timestamp.
        ledger.record("alice", "2026-06-16T11:00:00Z", -2.0, "later");
        ledger.record("alice", "2026-06-16T10:00:00Z", 5.0, "earlier");
        ledger.record("bob", "2026-06-16T10:30:00Z", 1.0, "other entity");

        let history = ledger.entries_for("alice");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].timestamp, "2026-06-16T10:00:00Z");
        assert_eq!(history[1].timestamp, "2026-06-16T11:00:00Z");
    }

    #[test]
    fn connection_entry_round_trips_through_json() {
        let entry = ConnectionEntry {
            entity_id: "alice".to_string(),
            timestamp: "2026-06-16T10:00:00Z".to_string(),
            change: 3.9,
            previous_balance: 0.0,
            new_balance: 3.9,
            notes: "warm greeting".to_string(),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: ConnectionEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, entry);
    }
}
