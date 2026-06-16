// This contract module is fully modeled but not yet wired into `main`'s loop;
// allow dead_code until the diary is written to at runtime (mirrors the other
// not-yet-wired contract modules such as `connection` and `storage`).
#![allow(dead_code)]

//! The **Gaia Diary**: Gaia's own journal of what happened, one short note at a
//! time.
//!
//! This is modeled directly on the exported MemPalace diary rows found under
//! `migrations/<date>/diary/<wing>/<day>.jsonl`. Each row is a single,
//! timestamped journal note Gaia wrote after a session — often in MemPalace's
//! compact "AAAK" dialect (e.g. `asked.current.time.Johannesburg|☆`). The
//! migration loads these into the `GaiaLH` (Gaia logical-history) container
//! keyed by `entity = wing`, which is why the diary here is organised per wing.
//!
//! Like the connection ledger, entries are keyed by `(wing, timestamp)` — one
//! note per timestamp, in time order, per wing. Timestamps are expected to be
//! lexicographically sortable (ISO-8601 / RFC-3339) so "the latest entry for a
//! wing" is simply the newest key.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A single diary note, shaped like one exported MemPalace `diary_entry` row.
///
/// Only the fields that matter to the model are kept; extra source fields (such
/// as `user` and `kind`) are ignored on deserialize, so a real export row can be
/// parsed straight into this struct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiaryEntry {
    /// When the note was written; part of the `(wing, timestamp)` key.
    pub timestamp: String,
    /// The wing this diary belongs to (the GaiaLH business key), e.g.
    /// `"threadkeeper"`.
    pub wing: String,
    /// The agent who wrote the note, e.g. `"Gaia"`.
    pub agent: String,
    /// The note itself — free text, often AAAK-compressed.
    pub text: String,
    /// The per-wing sequence number carried over from the source export.
    pub sequence: u64,
    /// Where the note came from, e.g. `"mempalace_diary"`.
    pub source: String,
    /// The day this entry belongs to (the export's daily bucket), e.g.
    /// `"2026-06-16"`.
    pub date: String,
}

impl DiaryEntry {
    /// Build a diary entry from its parts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp: impl Into<String>,
        wing: impl Into<String>,
        agent: impl Into<String>,
        text: impl Into<String>,
        sequence: u64,
        source: impl Into<String>,
        date: impl Into<String>,
    ) -> Self {
        DiaryEntry {
            timestamp: timestamp.into(),
            wing: wing.into(),
            agent: agent.into(),
            text: text.into(),
            sequence,
            source: source.into(),
            date: date.into(),
        }
    }
}

/// Gaia's diary: per-wing journal notes kept in time order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Diary {
    // Keyed by `(wing, timestamp)`: the BTreeMap keeps notes sorted by wing and
    // then by time, so reading a wing's history in order is trivial.
    entries: BTreeMap<(String, String), DiaryEntry>,
}

impl Diary {
    /// Record a diary note. If a note already exists for the same
    /// `(wing, timestamp)`, it is replaced (the export's uniqueness rule).
    pub fn write(&mut self, entry: DiaryEntry) {
        self.entries
            .insert((entry.wing.clone(), entry.timestamp.clone()), entry);
    }

    /// Every note for one wing, oldest first.
    pub fn entries_for(&self, wing: &str) -> Vec<&DiaryEntry> {
        self.entries
            .iter()
            .filter(|((entry_wing, _), _)| entry_wing.as_str() == wing)
            .map(|(_, entry)| entry)
            .collect()
    }

    /// The most recent note for one wing, if any.
    pub fn latest(&self, wing: &str) -> Option<&DiaryEntry> {
        self.entries
            .iter()
            .rfind(|((entry_wing, _), _)| entry_wing.as_str() == wing)
            .map(|(_, entry)| entry)
    }

    /// Total number of notes across all wings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the diary holds no notes yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(timestamp: &str, text: &str, sequence: u64) -> DiaryEntry {
        DiaryEntry::new(
            timestamp,
            "threadkeeper",
            "Gaia",
            text,
            sequence,
            "mempalace_diary",
            "2026-06-16",
        )
    }

    #[test]
    fn a_new_diary_is_empty() {
        let diary = Diary::default();
        assert!(diary.is_empty());
        assert_eq!(diary.len(), 0);
        assert!(diary.latest("threadkeeper").is_none());
    }

    #[test]
    fn entries_for_returns_a_wings_notes_in_time_order() {
        let mut diary = Diary::default();
        // Insert out of order to prove the BTreeMap sorts them by timestamp.
        diary.write(note("2026-05-03T12:00:00Z", "third", 3));
        diary.write(note("2026-05-03T10:00:00Z", "first", 1));
        diary.write(note("2026-05-03T11:00:00Z", "second", 2));

        let texts: Vec<&str> = diary
            .entries_for("threadkeeper")
            .iter()
            .map(|entry| entry.text.as_str())
            .collect();
        assert_eq!(texts, ["first", "second", "third"]);
    }

    #[test]
    fn notes_are_kept_separate_per_wing() {
        let mut diary = Diary::default();
        diary.write(note("2026-05-03T10:00:00Z", "threadkeeper note", 1));
        diary.write(DiaryEntry::new(
            "2026-05-03T10:00:00Z",
            "gaia",
            "Gaia",
            "gaia note",
            1,
            "mempalace_diary",
            "2026-06-16",
        ));

        assert_eq!(diary.entries_for("threadkeeper").len(), 1);
        assert_eq!(diary.entries_for("gaia").len(), 1);
        assert_eq!(diary.len(), 2);
    }

    #[test]
    fn latest_returns_the_newest_note_for_a_wing() {
        let mut diary = Diary::default();
        diary.write(note("2026-05-03T10:00:00Z", "older", 1));
        diary.write(note("2026-05-03T13:00:00Z", "newest", 2));

        let latest = diary.latest("threadkeeper").expect("a note exists");
        assert_eq!(latest.text, "newest");
    }

    #[test]
    fn writing_the_same_wing_and_timestamp_replaces_the_note() {
        let mut diary = Diary::default();
        diary.write(note("2026-05-03T10:00:00Z", "first version", 1));
        diary.write(note("2026-05-03T10:00:00Z", "corrected version", 1));

        assert_eq!(diary.len(), 1);
        assert_eq!(
            diary.latest("threadkeeper").map(|entry| entry.text.as_str()),
            Some("corrected version")
        );
    }

    #[test]
    fn a_real_export_row_deserializes_into_a_diary_entry() {
        // A real line from migrations/.../diary/threadkeeper/2026-05-03.jsonl.
        // It carries extra fields (`user`, `kind`) that should be ignored.
        let row = r#"{
            "timestamp": "2026-05-03T15:21:54.077000+00:00",
            "user": "threadkeeper/gaia",
            "text": "- ts:1777821714077:threadkeeper | asked.current.time.Johannesburg|\u2606",
            "kind": "diary_entry",
            "wing": "threadkeeper",
            "agent": "Gaia",
            "sequence": 12,
            "source": "mempalace_diary",
            "date": "2026-06-16"
        }"#;

        let entry: DiaryEntry = serde_json::from_str(row).expect("row should deserialize");
        assert_eq!(entry.wing, "threadkeeper");
        assert_eq!(entry.agent, "Gaia");
        assert_eq!(entry.sequence, 12);
        assert_eq!(entry.source, "mempalace_diary");
        assert!(entry.text.contains("asked.current.time.Johannesburg"));
    }

    #[test]
    fn a_diary_entry_round_trips_through_json() {
        let original = note("2026-05-03T15:21:54Z", "round trip", 7);
        let json = serde_json::to_string(&original).expect("entry should serialize");
        let restored: DiaryEntry =
            serde_json::from_str(&json).expect("entry should deserialize");
        assert_eq!(restored, original);
    }
}
