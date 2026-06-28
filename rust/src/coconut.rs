//! Coconut query-time pipeline — **rank** then **greedy pack**.
//!
//! At save time Gaia does the expensive work once per record (AAKKK
//! compression, token counting, embedding, and an LLM-assigned salience). This
//! module is the cheap query-time half described in `Coconut.md`:
//!
//! 1. [`rank`] — order candidate records by `final_rank = salience ×
//!    similarity`. **Similarity is computed in Cosmos, not here**: the DiskANN
//!    cosine vector search (`VectorDistance(...)`) runs in the database and
//!    projects a `similarityScore` onto each returned record (see
//!    `executor::plan_to_semantic_query`). This module simply reads that score
//!    the record and multiplies it by the memory's salience. Salience (a
//!    property of the *memory*, set by LLM Call 1 / Call 2) is kept strictly
//!    separate from similarity (a property of the *query–memory pair*), which
//!    makes the ranking easy to reason about and debug.
//! 2. [`pack`] — walk the ranked list top-to-bottom and greedily include each
//!    record whose token count still fits the budget, rendering every selected
//!    record the **same canonical way** (its AAKKK line) every time.
//!
//! [`coconut`] wires the two together. Every query logs its ranking and packing
//! decisions (step 9) so that when the output looks wrong, the truth is in the
//! logs first.

// The Coconut surface is being scaffolded incrementally; not every public item
// is wired into the engine yet. Mirrors `crate::storage` / `write_data_controller`.
#![allow(dead_code)]

use crate::storage::Record;

/// Approximate the token count of `text` with the same ~4-chars-per-token
/// heuristic used elsewhere in the codebase (`llm::approx_tokens`).
///
/// Coconut measures tokens on the exact text it intends to pack (the AAKKK
/// line), and uses the stored `token_count` when one was computed at save time;
/// this fallback keeps query-time packing correct for records written before a
/// count was stored.
pub fn count_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// The canonical text Coconut packs for a record: its dense AAKKK line when one
/// exists, otherwise the raw `data` as a fallback (so pre-Coconut records can
/// still be packed). Rendering the same way every time keeps output stable.
pub fn packing_text(record: &Record) -> &str {
    if record.aakkk.is_empty() {
        &record.data
    } else {
        &record.aakkk
    }
}

/// The number of tokens a record contributes when packed: the value measured at
/// save time when present, otherwise a fresh count of its [`packing_text`].
pub fn effective_token_count(record: &Record) -> usize {
    if record.token_count > 0 {
        record.token_count as usize
    } else {
        count_tokens(packing_text(record))
    }
}

/// One ranked candidate: the source record plus the two scores that ordered it.
///
/// Borrows the record (no clone) since ranking is a read-only scoring pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredRecord<'a> {
    /// The candidate record being scored.
    pub record: &'a Record,
    /// Cosine similarity of the query and record, as computed by Cosmos'
    /// DiskANN `VectorDistance(...)` search and read off the record's
    /// `similarity_score` field (a query–memory-pair property).
    pub similarity: f32,
    /// `salience × similarity` — the value the list is sorted by (descending).
    pub final_rank: f32,
}

/// The result of a greedy packing pass over a ranked list.
#[derive(Debug, Clone, PartialEq)]
pub struct PackResult {
    /// The canonical rendered context block: one AAKKK line per included record,
    /// newline-separated, in ranked order.
    pub rendered: String,
    /// The ids of the records that were included, in ranked order.
    pub included_ids: Vec<String>,
    /// Total tokens used by the included records (never exceeds the budget).
    pub used_tokens: usize,
    /// Sum of the included records' salience (a quick quality signal).
    pub total_salience: f32,
}

/// Rank candidate records by `final_rank` descending (Coconut step 4 / step 6).
///
/// The `records` are the rows a Cosmos semantic query already returned, each
/// carrying the `similarity_score` that the database's DiskANN cosine search
/// (`VectorDistance(...)`) computed for this query. This function does **not**
/// compute similarity itself — that work belongs in Cosmos. It only weights the
/// database's similarity by each memory's salience (`final_rank = salience ×
/// similarity`) and re-orders accordingly.
///
/// The ordering is **deterministic** for fixed inputs: ties on `final_rank` are
/// broken by ascending record id, so the same candidates always yield the same
/// order. Each record's score is logged (step 9).
pub fn rank(records: &[Record]) -> Vec<ScoredRecord<'_>> {
    let mut scored: Vec<ScoredRecord<'_>> = records
        .iter()
        .map(|record| {
            // Similarity comes from Cosmos (VectorDistance), not the backend.
            let similarity = record.similarity_score;
            let final_rank = record.salience * similarity;
            eprintln!(
                "coconut rank: id={} salience={:.4} similarity={:.4} final_rank={:.4}",
                record.record_id, record.salience, similarity, final_rank
            );
            ScoredRecord {
                record,
                similarity,
                final_rank,
            }
        })
        .collect();

    // Sort by final_rank DESC, breaking ties by id ASC for determinism. f32 has
    // no total order, so compare with partial_cmp and treat NaN as lowest.
    scored.sort_by(|a, b| {
        b.final_rank
            .partial_cmp(&a.final_rank)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.record_id.cmp(&b.record.record_id))
    });
    scored
}

/// Greedily pack the ranked list into `max_tokens` (Coconut step 7 / step 8).
///
/// Walks top-to-bottom; a record is included when its token count still fits the
/// remaining budget, otherwise it is **skipped** (and packing continues, so a
/// smaller lower-ranked record can still fill the tail of the budget). The
/// packed output never exceeds `max_tokens`. Every include/skip decision is
/// logged (step 9).
pub fn pack(scored: &[ScoredRecord<'_>], max_tokens: usize) -> PackResult {
    let mut rendered_lines: Vec<&str> = Vec::new();
    let mut included_ids: Vec<String> = Vec::new();
    let mut used_tokens = 0usize;
    let mut total_salience = 0.0f32;

    for item in scored {
        let tokens = effective_token_count(item.record);
        if used_tokens + tokens <= max_tokens {
            rendered_lines.push(packing_text(item.record));
            included_ids.push(item.record.record_id.clone());
            used_tokens += tokens;
            total_salience += item.record.salience;
            eprintln!(
                "coconut pack: id={} INCLUDED used_tokens={}",
                item.record.record_id, used_tokens
            );
        } else {
            eprintln!(
                "coconut pack: id={} SKIPPED (needs {} tokens, used_tokens={}, budget={})",
                item.record.record_id, tokens, used_tokens, max_tokens
            );
        }
    }

    PackResult {
        rendered: rendered_lines.join("\n"),
        included_ids,
        used_tokens,
        total_salience,
    }
}

/// End-to-end Coconut query: rank the Cosmos-returned candidates by `salience ×
/// similarity`, then greedily pack the most useful records into `max_tokens`.
///
/// `records` are the rows a semantic query already returned, each carrying the
/// `similarity_score` Cosmos computed via DiskANN. Returns the canonical packed
/// context block ready to hand to the LLM.
pub fn coconut(records: &[Record], max_tokens: usize) -> PackResult {
    let scored = rank(records);
    pack(&scored, max_tokens)
}

/// Build the canonical save-time AAKKK line for a record **and** its token
/// count, so both can be stored once at save time (Coconut steps 2, 3, 5).
///
/// The line carries the entity, a fluff-stripped compression of the text, the
/// `salience` (when set), and the timestamp. The token count is measured on the
/// exact line that will later be packed — deliberately **without** embedding a
/// `K:tokens=` field, which would be self-referential (the count depends on the
/// line, and the line would depend on the count). The count therefore always
/// equals `count_tokens(line)`, keeping save-time and query-time in agreement.
///
/// `salience` is `None` when no LLM salience is available yet (it is then
/// omitted from the line); otherwise it is rendered to four decimals.
pub fn build_save_aakkk(
    entity: &str,
    text: &str,
    salience: Option<f32>,
    timestamp: &str,
) -> (String, u32) {
    let compressed = crate::aakkk::strip_fluff(text);
    let salience_field = salience.map(|s| format!("{s:.4}")).unwrap_or_default();
    let line = crate::aakkk::AakkkLine::new()
        .attr("entity", entity)
        .attr("text", &compressed)
        .key("salience", &salience_field)
        .key("ts", timestamp)
        .render();
    let tokens = count_tokens(&line) as u32;
    (line, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::RecordKind;

    /// Build a KB record carrying a Cosmos-style `similarity_score`, an AAKKK
    /// line, a token count and a salience — the fields Coconut ranks and packs
    /// on. The similarity stands in for what `VectorDistance(...)` would project.
    fn record(id: &str, similarity: f32, aakkk: &str, tokens: u32, salience: f32) -> Record {
        let mut r = Record::new(
            id,
            "rust",
            "",
            "2026-06-28",
            RecordKind::KnowledgeBase,
            "raw",
            Vec::new(),
        )
        .with_coconut_fields(aakkk, tokens, salience, "t0", "t1");
        r.similarity_score = similarity;
        r
    }

    #[test]
    fn count_tokens_rounds_up_quarter_of_chars() {
        assert_eq!(count_tokens(""), 0);
        assert_eq!(count_tokens("abc"), 1);
        assert_eq!(count_tokens("abcd"), 1);
        assert_eq!(count_tokens("abcde"), 2);
    }

    #[test]
    fn rank_orders_by_salience_times_similarity_descending() {
        let records = vec![
            // High similarity, low salience -> mid rank.
            record("a", 1.0, "A:e=a", 1, 0.2),
            // Mid similarity, high salience -> top rank.
            record("b", 0.707, "A:e=b", 1, 0.9),
            // Zero similarity -> zero final_rank.
            record("c", 0.0, "A:e=c", 1, 1.0),
        ];

        let scored = rank(&records);
        let order: Vec<&str> = scored.iter().map(|s| s.record.record_id.as_str()).collect();
        // b: 0.9*0.707=0.636, a: 0.2*1.0=0.2, c: 1.0*0=0.0
        assert_eq!(order, vec!["b", "a", "c"]);
    }

    #[test]
    fn rank_is_deterministic_with_id_tiebreak() {
        // Two records with identical scores: tie broken by ascending id.
        let records = vec![
            record("z", 1.0, "A:e=z", 1, 0.5),
            record("a", 1.0, "A:e=a", 1, 0.5),
        ];
        let scored = rank(&records);
        let order: Vec<&str> = scored.iter().map(|s| s.record.record_id.as_str()).collect();
        assert_eq!(order, vec!["a", "z"]);
    }

    #[test]
    fn pack_greedily_fills_the_budget_and_never_exceeds_it() {
        let records = vec![
            record("a", 1.0, "A:e=a", 3, 0.9), // top rank, 3 tokens
            record("b", 1.0, "A:e=b", 5, 0.8), // 5 tokens, would overflow
            record("c", 1.0, "A:e=c", 2, 0.7), // 2 tokens, fits the tail
        ];
        let scored = rank(&records);
        let result = pack(&scored, 5);

        // a (3) fits, b (5) overflows and is skipped, c (2) fits -> total 5.
        assert_eq!(result.included_ids, vec!["a".to_string(), "c".to_string()]);
        assert_eq!(result.used_tokens, 5);
        assert!(result.used_tokens <= 5);
        assert_eq!(result.rendered, "A:e=a\nA:e=c");
        assert!((result.total_salience - 1.6).abs() < 1e-6);
    }

    #[test]
    fn pack_falls_back_to_counting_when_no_token_count_is_stored() {
        // A pre-Coconut record: no aakkk, no stored token_count -> packs the raw
        // data and counts its tokens on the fly.
        let bare = Record::new(
            "bare",
            "rust",
            "",
            "2026-06-28",
            RecordKind::KnowledgeBase,
            "abcdefgh", // 8 chars -> 2 tokens
            Vec::new(),
        );
        let mut bare = bare;
        bare.salience = 0.5;
        bare.similarity_score = 1.0;
        let records = vec![bare];
        let scored = rank(&records);
        let result = pack(&scored, 2);
        assert_eq!(result.included_ids, vec!["bare".to_string()]);
        assert_eq!(result.used_tokens, 2);
        assert_eq!(result.rendered, "abcdefgh");
    }

    #[test]
    fn coconut_end_to_end_returns_a_canonical_block() {
        let records = vec![
            record("a", 1.0, "A:e=a|K:tokens=1", 1, 0.9),
            record("b", 0.0, "A:e=b|K:tokens=1", 1, 0.9),
        ];
        let result = coconut(&records, 100);
        // a is similar to the query, b is not -> a ranks first; both fit.
        assert_eq!(result.included_ids[0], "a");
        assert!(result.rendered.starts_with("A:e=a"));
    }

    #[test]
    fn pack_with_zero_budget_includes_nothing() {
        let records = vec![record("a", 1.0, "A:e=a", 1, 0.9)];
        let scored = rank(&records);
        let result = pack(&scored, 0);
        assert!(result.included_ids.is_empty());
        assert_eq!(result.used_tokens, 0);
        assert_eq!(result.rendered, "");
    }

    #[test]
    fn build_save_aakkk_includes_salience_and_matches_its_token_count() {
        let (line, tokens) =
            build_save_aakkk("rust", "the quick brown fox", Some(0.875), "20260628T0458");
        // Entity + fluff-stripped text + salience + ts, no self-referential tokens.
        assert!(line.starts_with("A:entity=rust|A:text=quick brown fox"));
        assert!(line.contains("K:salience=0.8750"));
        assert!(line.contains("K:ts=20260628T0458"));
        assert!(!line.contains("K:tokens="));
        // The stored count equals a fresh count of the exact stored line.
        assert_eq!(tokens as usize, count_tokens(&line));
    }

    #[test]
    fn build_save_aakkk_omits_salience_when_unset() {
        let (line, _tokens) = build_save_aakkk("rust", "a plain note", None, "20260628T0458");
        assert!(!line.contains("K:salience="));
        assert!(line.contains("K:ts=20260628T0458"));
    }

    #[test]
    fn build_save_aakkk_is_deterministic() {
        let a = build_save_aakkk("rust", "same text here", Some(0.5), "t");
        let b = build_save_aakkk("rust", "same text here", Some(0.5), "t");
        assert_eq!(a, b);
    }
}
