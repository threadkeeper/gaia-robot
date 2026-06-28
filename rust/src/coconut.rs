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
//! [`coconut`] wires the two together.

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
/// order.
pub fn rank(records: &[Record]) -> Vec<ScoredRecord<'_>> {
    let mut scored: Vec<ScoredRecord<'_>> = records
        .iter()
        .map(|record| {
            // Similarity comes from Cosmos (VectorDistance), not the backend.
            let similarity = record.similarity_score;
            let final_rank = record.salience * similarity;
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
/// packed output never exceeds `max_tokens`.
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

/// The canonical AAKKK line for a record: the stored `aakkk` field when present,
/// otherwise one rebuilt on the fly from the record's fields (so pre-Coconut KB
/// rows, which never stored an `aakkk`, still produce a knowledge line).
///
/// The rebuilt line uses the record's `salience` (omitted when zero), and the
/// freshest available timestamp (`updated_at`, else `created_at`, else `date`).
fn aakkk_line_for(record: &Record) -> String {
    if !record.aakkk.is_empty() {
        return record.aakkk.clone();
    }
    // A zero salience means "none assigned", so drop it from the rebuilt line.
    let salience = (record.salience != 0.0).then_some(record.salience);
    let timestamp = if !record.updated_at.is_empty() {
        record.updated_at.as_str()
    } else if !record.created_at.is_empty() {
        record.created_at.as_str()
    } else {
        record.date.as_str()
    };
    build_save_aakkk(&record.entity_id, &record.data, salience, timestamp).0
}

/// The **knowledge part** of a record for a KB summary: its AAKKK line with the
/// knowledge-free `A:entity=…` attribute and the `A:text=` field label stripped,
/// leaving just the (escaped) knowledge text followed by its `K:` metadata keys.
///
/// For example the stored line
/// `A:entity=github:259026842|A:text=Jonty -> status -> sleepy|K:salience=0.2000|K:ts=2026-06-27T17:44:15Z`
/// renders as
/// `Jonty -> status -> sleepy|K:salience=0.2000|K:ts=2026-06-27T17:44:15Z`.
///
/// The entity is the same for every record in one account's KB and carries no
/// knowledge, so it is dropped; the text value keeps its `\|` escaping so the
/// summary stays a parseable AAKKK fragment.
pub fn knowledge_part(record: &Record) -> String {
    let line = aakkk_line_for(record);
    // Drop a leading `A:entity=…|` attribute. Entity values never contain a raw
    // `|`, so the first unescaped `|` ends the attribute.
    let without_entity = match line.split_once('|') {
        Some((head, rest)) if head.starts_with("A:entity=") => rest,
        _ => line.as_str(),
    };
    // Drop the `A:text=` label, keeping just its (still-escaped) value + K: keys.
    without_entity
        .strip_prefix("A:text=")
        .unwrap_or(without_entity)
        .to_string()
}

/// Pack an account's most important KB records into one pipe-delimited knowledge
/// summary that stays within `max_token_total` tokens.
///
/// This is the query-less KB-summary form of Coconut. The full formula is
/// `final_rank = salience × similarity`, but with **no query** there is no
/// similarity term, so ranking reduces to **salience descending** (tie-broken by
/// id ascending for determinism, matching [`rank`]). Walking that order
/// top-to-bottom, each record's [`knowledge_part`] is included while the running
/// token total would still stay at or below `max_token_total`; a record that
/// would exceed the budget is skipped and packing continues (a smaller, lower
/// record can still fill the tail). Tokens are measured on the exact text packed
/// (the knowledge part), so the count reflects what the summary actually emits.
///
/// The included parts are joined with `|` into [`PackResult::rendered`], the
/// same struct [`pack`] returns, so callers get the rendered block plus the
/// included ids, the used token count, and the summed salience.
pub fn knowledge_summary(records: &[Record], max_token_total: usize) -> PackResult {
    // Rank by salience DESC, id ASC. (Not `rank`, which multiplies by the
    // per-query similarity score — that is zero here with no query and would
    // flatten the ordering.)
    let mut ordered: Vec<&Record> = records.iter().collect();
    ordered.sort_by(|a, b| {
        b.salience
            .partial_cmp(&a.salience)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record_id.cmp(&b.record_id))
    });

    let mut parts: Vec<String> = Vec::new();
    let mut included_ids: Vec<String> = Vec::new();
    let mut used_tokens = 0usize;
    let mut total_salience = 0.0f32;

    for record in ordered {
        let part = knowledge_part(record);
        let tokens = count_tokens(&part);
        if used_tokens + tokens <= max_token_total {
            used_tokens += tokens;
            total_salience += record.salience;
            included_ids.push(record.record_id.clone());
            parts.push(part);
        }
    }

    PackResult {
        rendered: parts.join("|"),
        included_ids,
        used_tokens,
        total_salience,
    }
}

/// The per-turn token budget for the GaiaKB knowledge handed to LLM Call 2 —
/// the "threshold allowed for KB" in the Coconut design.
///
/// Gaia's full grounding context drawn from Cosmos is large (see `flow.rs`), but
/// the knowledge base is only one of several sources feeding LLM Call 2, so its
/// share is capped here. The cap keeps the most salient, most relevant facts
/// while bounding how much of the context window the KB can consume as the
/// account's knowledge grows. Tune this as the overall context budget is
/// formalised.
pub const KB_CONTEXT_TOKEN_BUDGET: usize = 2048;

/// Reduce the GaiaKB rows retrieved this turn to the subset LLM Call 2 should
/// see: rank by the Coconut formula and keep the highest-ranked records that
/// fit `max_token_total`, returned (cloned) in ranked order.
///
/// This is the query-time wiring of Coconut into the pull → push hand-off. The
/// full formula is `final_rank = salience × similarity`: when the KB query ran
/// semantically, Cosmos has projected a `similarity_score` onto every row, so
/// the product orders them. When no similarity is present (a keyword / most-
/// recent KB query — e.g. the forced core-user query), every product would
/// collapse to zero and flatten the order, so ranking falls back to **salience
/// descending** instead (matching [`knowledge_summary`]). Ties are broken by
/// ascending record id for determinism.
///
/// Walking that order top-to-bottom, each record is kept while the running token
/// total (measured with [`effective_token_count`]) still fits `max_token_total`;
/// a record that would overflow is skipped and packing continues, so a smaller
/// lower-ranked record can still fill the tail of the budget.
pub fn kb_context_records(records: &[Record], max_token_total: usize) -> Vec<Record> {
    // Use salience × similarity only when Cosmos projected real similarities this
    // turn; otherwise rank by salience alone so the order stays meaningful.
    let has_similarity = records.iter().any(|record| record.similarity_score != 0.0);
    let score = |record: &&Record| -> f32 {
        if has_similarity {
            record.salience * record.similarity_score
        } else {
            record.salience
        }
    };

    let mut ordered: Vec<&Record> = records.iter().collect();
    ordered.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record_id.cmp(&b.record_id))
    });

    let mut selected: Vec<Record> = Vec::new();
    let mut used_tokens = 0usize;
    for record in ordered {
        let tokens = effective_token_count(record);
        if used_tokens + tokens <= max_token_total {
            used_tokens += tokens;
            selected.push(record.clone());
        }
    }
    selected
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

    #[test]
    fn knowledge_part_strips_entity_and_text_label_keeping_metadata() {
        // Mirrors the live KB shape: entity attr + escaped text + K: keys.
        let line = "A:entity=github:259026842|A:text=Jonty -> status -> sleepy|K:salience=0.2000|K:ts=2026-06-27T17:44:15Z";
        let r = record("id1", 0.0, line, 0, 0.2);
        assert_eq!(
            knowledge_part(&r),
            "Jonty -> status -> sleepy|K:salience=0.2000|K:ts=2026-06-27T17:44:15Z"
        );
    }

    #[test]
    fn knowledge_part_preserves_escaped_inner_pipes_in_the_text() {
        // The text value carries `\|`-escaped internal pipes; those must survive.
        let line = "A:entity=e|A:text=a \\| b \\| c|K:salience=0.5000|K:ts=t";
        let r = record("id2", 0.0, line, 0, 0.5);
        assert_eq!(knowledge_part(&r), "a \\| b \\| c|K:salience=0.5000|K:ts=t");
    }

    #[test]
    fn knowledge_part_rebuilds_when_no_aakkk_is_stored() {
        // A pre-Coconut record (no aakkk): the line is rebuilt from its fields,
        // then the entity + A:text= label are stripped the same way.
        let mut r = Record::new(
            "id3",
            "Jonty",
            "",
            "2026-06-16",
            RecordKind::KnowledgeBase,
            "Jonty is an engineer",
            Vec::new(),
        );
        r.salience = 0.8;
        let part = knowledge_part(&r);
        // Entity dropped, "is"/"an" filler stripped from the text, salience kept.
        assert_eq!(part, "Jonty engineer|K:salience=0.8000|K:ts=2026-06-16");
    }

    #[test]
    fn knowledge_summary_packs_by_salience_within_the_token_budget() {
        // Three records of differing salience; salience decides inclusion order.
        let high = record("a", 0.0, "A:entity=e|A:text=high", 0, 0.9);
        let mid = record("b", 0.0, "A:entity=e|A:text=mid", 0, 0.5);
        let low = record("c", 0.0, "A:entity=e|A:text=low", 0, 0.1);
        let records = vec![low, high, mid]; // unsorted on purpose

        let high_part = knowledge_part(&records[1]);
        let mid_part = knowledge_part(&records[2]);
        // Budget for exactly the top two by salience (high then mid).
        let budget = count_tokens(&high_part) + count_tokens(&mid_part);
        let summary = knowledge_summary(&records, budget);

        assert_eq!(summary.included_ids, vec!["a".to_string(), "b".to_string()]);
        assert!(summary.used_tokens <= budget);
        // Rendered is the two knowledge parts joined by a single `|`.
        assert_eq!(summary.rendered, format!("{high_part}|{mid_part}"));
        assert!((summary.total_salience - 1.4).abs() < 1e-6);
    }

    #[test]
    fn knowledge_summary_skips_an_oversized_record_and_keeps_filling() {
        let big = record(
            "a",
            0.0,
            "A:entity=e|A:text=this is a much longer knowledge fact about something",
            0,
            0.9,
        );
        let small = record("b", 0.0, "A:entity=e|A:text=short", 0, 0.8);
        let records = vec![big, small];

        let small_part = knowledge_part(&records[1]);
        let small_tokens = count_tokens(&small_part);
        // Budget fits only the small record; the bigger top-salience one is skipped.
        let summary = knowledge_summary(&records, small_tokens);

        assert_eq!(summary.included_ids, vec!["b".to_string()]);
        assert_eq!(summary.used_tokens, small_tokens);
        assert_eq!(summary.rendered, small_part);
    }

    #[test]
    fn knowledge_summary_with_zero_budget_is_empty() {
        let records = vec![record("a", 0.0, "A:entity=e|A:text=anything", 0, 0.9)];
        let summary = knowledge_summary(&records, 0);
        assert!(summary.included_ids.is_empty());
        assert_eq!(summary.used_tokens, 0);
        assert_eq!(summary.rendered, "");
    }

    #[test]
    fn knowledge_summary_is_deterministic_with_id_tiebreak() {
        // Equal salience -> ordered by ascending id, so "a" precedes "z".
        let z = record("z", 0.0, "A:entity=e|A:text=zeta", 0, 0.5);
        let a = record("a", 0.0, "A:entity=e|A:text=alpha", 0, 0.5);
        let records = vec![z, a];
        let summary = knowledge_summary(&records, 1000);
        assert_eq!(summary.included_ids, vec!["a".to_string(), "z".to_string()]);
    }

    #[test]
    fn kb_context_records_ranks_by_salience_times_similarity_when_present() {
        // With Cosmos-projected similarities, the full Coconut formula orders the
        // rows: b (0.9*0.707) > a (0.2*1.0) > c (1.0*0.0).
        let records = vec![
            record("a", 1.0, "A:e=a", 1, 0.2),
            record("b", 0.707, "A:e=b", 1, 0.9),
            record("c", 0.0, "A:e=c", 1, 1.0),
        ];
        let selected = kb_context_records(&records, 1000);
        let order: Vec<&str> = selected.iter().map(|r| r.record_id.as_str()).collect();
        assert_eq!(order, vec!["b", "a", "c"]);
    }

    #[test]
    fn kb_context_records_falls_back_to_salience_when_no_similarity() {
        // A keyword / most-recent query projects no similarity (all zero), so the
        // product would flatten the order; ranking must fall back to salience.
        let records = vec![
            record("low", 0.0, "A:e=low", 1, 0.1),
            record("high", 0.0, "A:e=high", 1, 0.9),
            record("mid", 0.0, "A:e=mid", 1, 0.5),
        ];
        let selected = kb_context_records(&records, 1000);
        let order: Vec<&str> = selected.iter().map(|r| r.record_id.as_str()).collect();
        assert_eq!(order, vec!["high", "mid", "low"]);
    }

    #[test]
    fn kb_context_records_caps_at_the_token_budget_and_keeps_filling() {
        // Ranked a(3), b(5), c(2) by salience; budget 5 keeps a then skips the
        // oversized b and still fits c in the tail -> total 5 tokens, never over.
        let records = vec![
            record("a", 0.0, "A:e=a", 3, 0.9),
            record("b", 0.0, "A:e=b", 5, 0.8),
            record("c", 0.0, "A:e=c", 2, 0.7),
        ];
        let selected = kb_context_records(&records, 5);
        let order: Vec<&str> = selected.iter().map(|r| r.record_id.as_str()).collect();
        assert_eq!(order, vec!["a", "c"]);
        let used: usize = selected.iter().map(effective_token_count).sum();
        assert_eq!(used, 5);
    }

    #[test]
    fn kb_context_records_with_zero_budget_selects_nothing() {
        let records = vec![record("a", 1.0, "A:e=a", 1, 0.9)];
        assert!(kb_context_records(&records, 0).is_empty());
    }
}
