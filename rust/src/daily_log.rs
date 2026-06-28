//! The **daily conversation log** stored in `GaiaDataLake`: a merge-friendly
//! JSON document that accumulates a day's conversation turns without ever
//! duplicating one.
//!
//! Every turn that lands in `GaiaDataLake` is appended to a single per-day
//! record (`GaiaDataLake|{entity}|{YYYY-MM-DD}`). Rather than storing that day
//! as one flat string, the record's `data` field holds a small JSON object:
//!
//! ```json
//! {
//!   "turns": [
//!     { "ts": "04:58:00Z", "key": "<sha1 of text>", "text": "User: hi\nGaia: hello" }
//!   ]
//! }
//! ```
//!
//! Each turn carries a stable content `key` (a SHA-1 of its text). Merging a new
//! turn is therefore trivial and idempotent: if a turn with the same key already
//! exists, the merge is a no-op, so re-sending the same turn (a retry, a replay,
//! a duplicated request) can never duplicate it within the day.
//!
//! The JSON is the *storage* form. For embedding and for the Response Data
//! Context shown to the LLM, the log renders back to a plain-text transcript via
//! [`DailyLog::to_transcript`], so vectors and grounding text stay clean,
//! human-readable conversation rather than JSON syntax.

// The daily-log surface is being scaffolded incrementally; the parse/merge/
// to_json helpers are not wired into the write path yet. Mirrors the
// `#![allow(dead_code)]` already used in `crate::storage` / `write_data_controller`.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

/// One conversation turn within a day's log.
///
/// `ts` is the time-of-day marker (e.g. `04:58:00Z`); `key` is the stable
/// content hash used to deduplicate; `text` is the verbatim turn body
/// (typically `"User: …\nGaia: …"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DailyTurn {
    /// The time-of-day marker for this turn (e.g. `04:58:00Z`). May be empty for
    /// a legacy record that was migrated from the old flat-text format.
    ts: String,
    /// A stable SHA-1 (hex) of `text`, used purely as a dedup key.
    key: String,
    /// The verbatim turn text, e.g. `"User: hi\nGaia: hello"`.
    text: String,
}

/// A day's worth of conversation turns, in arrival order.
///
/// Build one from a stored record with [`DailyLog::parse`], add a turn with
/// [`DailyLog::merge`], then serialize back with [`DailyLog::to_json`] (for
/// storage) and [`DailyLog::to_transcript`] (for embedding / display).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyLog {
    /// The day's turns, oldest first.
    turns: Vec<DailyTurn>,
}

impl DailyLog {
    /// Parse a stored `data` body into a [`DailyLog`].
    ///
    /// Three cases are handled so the parser never fails:
    /// - empty / whitespace-only input yields an empty log;
    /// - a JSON document in this module's shape is parsed as-is;
    /// - anything else (a legacy flat-text day from before this format existed)
    ///   is wrapped losslessly as a single turn, so old records migrate forward
    ///   the first time they are written again rather than being discarded.
    pub fn parse(data: &str) -> Self {
        let trimmed = data.trim();
        if trimmed.is_empty() {
            return Self::default();
        }
        // Preferred path: the data is already a JSON daily log.
        if let Ok(log) = serde_json::from_str::<DailyLog>(trimmed) {
            return log;
        }
        // Legacy migration: treat the whole prior body as one existing turn so
        // nothing is lost. Its key is the hash of the legacy text, so a brand-new
        // turn (with different text) will still append cleanly.
        Self {
            turns: vec![DailyTurn {
                ts: String::new(),
                key: content_key(data),
                text: data.to_string(),
            }],
        }
    }

    /// Merge one new turn into the log, returning `true` when it was added and
    /// `false` when it was a duplicate (already present, so nothing changed).
    ///
    /// Dedup is by the turn's content `key`, so the same `text` is never stored
    /// twice in a day no matter how many times it is submitted.
    pub fn merge(&mut self, time_marker: &str, text: &str) -> bool {
        let key = content_key(text);
        if self.turns.iter().any(|turn| turn.key == key) {
            return false;
        }
        self.turns.push(DailyTurn {
            ts: time_marker.to_string(),
            key,
            text: text.to_string(),
        });
        true
    }

    /// Serialize the log to its canonical JSON storage form.
    ///
    /// Serialization is deterministic (fixed field order, turns in arrival
    /// order), so an unchanged log always produces byte-identical JSON — which
    /// keeps the content-hash replay guard stable.
    pub fn to_json(&self) -> String {
        // Serializing a plain struct of strings cannot fail; fall back to an
        // empty object on the impossible error rather than panicking.
        serde_json::to_string(self).unwrap_or_else(|_| "{\"turns\":[]}".to_string())
    }

    /// Render the log as a plain-text transcript for embedding and display.
    ///
    /// Each turn becomes `"[<ts>] <text>"` (or just `"<text>"` when a migrated
    /// legacy turn has no timestamp), joined by blank lines. This matches the
    /// readable, chronologically-ordered shape the flat-text format used, so the
    /// embedding and the Response Data Context see clean conversation text.
    pub fn to_transcript(&self) -> String {
        let mut out = String::new();
        for turn in &self.turns {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            if turn.ts.is_empty() {
                out.push_str(&turn.text);
            } else {
                // Writing to a String is infallible, so the result is ignored.
                let _ = write!(out, "[{}] {}", turn.ts, turn.text);
            }
        }
        out
    }
}

/// Render a stored `data` body to a transcript **iff** it is a JSON daily log.
///
/// Retrieval reads the raw `data` field straight from Cosmos. For `GaiaDataLake`
/// that field is now a JSON daily log, so this lets the response-context renderer
/// surface the readable transcript instead of leaking JSON. Returns `None` for
/// any body that is not a JSON daily log (e.g. a plain-text diary note), leaving
/// it untouched.
pub fn transcript_if_daily_log(data: &str) -> Option<String> {
    let trimmed = data.trim();
    // Only treat genuine JSON objects as candidates; a plain-text body must be
    // left exactly as it is.
    if !trimmed.starts_with('{') {
        return None;
    }
    let log = serde_json::from_str::<DailyLog>(trimmed).ok()?;
    Some(log.to_transcript())
}

/// Compute a stable hex SHA-1 of `text`, used only as a dedup key (never for
/// security). Identical text always yields the same key.
fn content_key(text: &str) -> String {
    let digest = crate::sha1::digest(text.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        // Writing to a String is infallible, so the result can be ignored.
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_yields_an_empty_log() {
        assert_eq!(DailyLog::parse(""), DailyLog::default());
        assert_eq!(DailyLog::parse("   \n  "), DailyLog::default());
    }

    #[test]
    fn merge_adds_a_new_turn_and_reports_change() {
        let mut log = DailyLog::default();
        let added = log.merge("09:00:00Z", "User: hi\nGaia: hello");
        assert!(added, "a brand-new turn is added");
        assert_eq!(log.turns.len(), 1);
    }

    #[test]
    fn merge_never_duplicates_an_identical_turn() {
        let mut log = DailyLog::default();
        assert!(log.merge("09:00:00Z", "User: hi\nGaia: hello"));
        // Same text again (even at a different time) is a duplicate: no change.
        let added_again = log.merge("10:00:00Z", "User: hi\nGaia: hello");
        assert!(!added_again, "an identical turn must not be added twice");
        assert_eq!(log.turns.len(), 1, "the day still holds exactly one turn");
    }

    #[test]
    fn merge_keeps_distinct_turns() {
        let mut log = DailyLog::default();
        assert!(log.merge("09:00:00Z", "User: hi\nGaia: hello"));
        assert!(log.merge("09:05:00Z", "User: bye\nGaia: see you"));
        assert_eq!(log.turns.len(), 2);
    }

    #[test]
    fn json_round_trips_through_parse() {
        let mut log = DailyLog::default();
        log.merge("09:00:00Z", "User: hi\nGaia: hello");
        log.merge("09:05:00Z", "User: bye\nGaia: see you");
        let json = log.to_json();
        assert_eq!(DailyLog::parse(&json), log, "JSON parse round-trips");
        // The canonical JSON carries the turns array.
        assert!(json.contains("\"turns\""));
    }

    #[test]
    fn legacy_flat_text_is_migrated_as_one_turn() {
        let legacy = "[08:00:00Z] User: earlier\nGaia: note";
        let log = DailyLog::parse(legacy);
        assert_eq!(log.turns.len(), 1, "legacy text becomes a single turn");
        // A new, different turn still appends cleanly on top of the migrated one.
        let mut log = log;
        assert!(log.merge("09:00:00Z", "User: now\nGaia: reply"));
        assert_eq!(log.turns.len(), 2);
    }

    #[test]
    fn transcript_renders_readable_conversation() {
        let mut log = DailyLog::default();
        log.merge("09:00:00Z", "User: hi\nGaia: hello");
        log.merge("09:05:00Z", "User: bye\nGaia: see you");
        let transcript = log.to_transcript();
        assert_eq!(
            transcript,
            "[09:00:00Z] User: hi\nGaia: hello\n\n[09:05:00Z] User: bye\nGaia: see you"
        );
    }

    #[test]
    fn transcript_if_daily_log_only_matches_json() {
        let mut log = DailyLog::default();
        log.merge("09:00:00Z", "User: hi\nGaia: hello");
        let json = log.to_json();
        assert_eq!(
            transcript_if_daily_log(&json).as_deref(),
            Some("[09:00:00Z] User: hi\nGaia: hello")
        );
        // A plain-text body is left untouched (None).
        assert_eq!(transcript_if_daily_log("[09:00:00Z] just text"), None);
    }
}
