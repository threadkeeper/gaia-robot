//! Deterministic assembly of the **Response Data Context** that bridges LLM
//! Call 1 and LLM Call 2.
//!
//! In the Gaia physical architecture, LLM Call 1 produces four documents
//! (`actions.json`, `analysis.json`, `facts.json`, `newContext.json`) and its
//! actions drive retrieval against the Web and the Cosmos containers. All of
//! that gathered material has to be folded into a single grounding document
//! that LLM Call 2 reads. This module builds that document **without** making
//! another LLM call: it is pure string assembly over Call 1's output plus the
//! retrieval results, so the same inputs always yield byte-identical markdown.
//!
//! The document always contains these eight headings, in this exact order:
//! `WebSearchResults`, `DataLakeResults`, `KnowledgeBaseResults`,
//! `ConnectionsResults`, `EmotionResults`, `TruthfulNessResults`,
//! `IntentionResults`, `OldContextSummary`. Retrieval containers that have no
//! dedicated heading (the diary, the conversation data lake) are folded into
//! `DataLakeResults`, and `facts.json` is folded into `KnowledgeBaseResults`,
//! so nothing Call 1 gathered is ever dropped.

use serde::Deserialize;
use serde_json::{Map, Value};
use std::fmt::Write as _;

/// LLM Call 1's `analysis.json` (array element 1): the model's read on the
/// user's emotional state, truthfulness, and intention for this turn.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct TurnAnalysis {
    /// Free-form description of the user's emotional state.
    #[serde(default)]
    pub emotion: String,
    /// Free-form assessment of how truthful the user appears to be.
    #[serde(default)]
    pub truthfulness: String,
    /// Free-form description of what the user is trying to achieve.
    #[serde(default)]
    pub intention: String,
}

/// One durable fact from LLM Call 1's `facts.json` (array element 2).
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct TurnFact {
    /// A short key naming the fact (e.g. `"favourite_colour"`).
    #[serde(default)]
    pub fact: String,
    /// The fact's value (e.g. `"blue"`).
    #[serde(default)]
    pub value: String,
}

/// Everything Call 1 produced *besides* the action plan, parsed once from the
/// raw reply so it can be rendered into the Response Data Context.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Call1Extras {
    /// The `analysis.json` document (element 1).
    pub analysis: TurnAnalysis,
    /// The `facts.json` document (element 2).
    pub facts: Vec<TurnFact>,
    /// The `summary` field of the `newContext.json` document (element 3).
    pub old_context_summary: String,
}

/// One action's retrieval results: the records a single planned query or web
/// search returned, tagged with the action that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalGroup {
    /// The originating action's id (e.g. `"q3"`).
    pub action_id: String,
    /// The container or source the records came from (e.g. `"GaiaKB"`,
    /// `"Web"`, `"GaiaConnections"`).
    pub container: String,
    /// The raw records returned, as JSON values.
    pub records: Vec<Value>,
}

/// The four retrieval-backed sections of the Response Data Context. Every
/// container maps to exactly one of these so nothing is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetrievalSection {
    /// `WebSearchResults` — anything sourced from the web.
    WebSearch,
    /// `DataLakeResults` — the conversation data lake and the diary.
    DataLake,
    /// `KnowledgeBaseResults` — the durable knowledge base.
    KnowledgeBase,
    /// `ConnectionsResults` — the emotional-bank-account ledger.
    Connections,
}

/// Parse Call 1's non-action documents (analysis, facts, newContext) out of a
/// raw reply. Missing or malformed documents degrade gracefully to defaults so
/// the Response Data Context can always be produced.
pub fn parse_call1_extras(reply: &str) -> Call1Extras {
    // Reuse the shared extractor so we agree with the actions parser on exactly
    // which `[...]` block is Call 1's output array.
    let array = match crate::actions::extract_call1_array(reply) {
        Some(array) => array,
        None => return Call1Extras::default(),
    };

    // Element 1 is analysis.json; element 2 is facts.json; element 3 is
    // newContext.json. Any of them may be absent in a degraded reply.
    let analysis = array
        .get(1)
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default();
    let facts = array
        .get(2)
        .and_then(|value| serde_json::from_value::<Vec<TurnFact>>(value.clone()).ok())
        .unwrap_or_default();
    let old_context_summary = array
        .get(3)
        .and_then(|value| value.get("summary"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();

    Call1Extras {
        analysis,
        facts,
        old_context_summary,
    }
}

/// Build the full Response Data Context markdown handed to LLM Call 2.
///
/// The output is deterministic: identical inputs always yield identical bytes.
/// All eight required headings are emitted in order even when the underlying
/// data is empty, so Call 2 sees a stable, predictable structure.
pub fn build_response_data_context(
    user_id: &str,
    question: &str,
    requested_at: &str,
    extras: &Call1Extras,
    groups: &[RetrievalGroup],
) -> String {
    let mut md = String::new();

    // Preamble: what this document is and the turn it describes.
    let _ = writeln!(md, "# Response Data Context\n");
    let _ = writeln!(
        md,
        "Deterministically assembled between LLM Call 1 and LLM Call 2 from everything \
         Call 1 gathered this turn. This is the sole grounding context handed to LLM Call 2.\n"
    );
    let _ = writeln!(md, "- **user_id:** {user_id}");
    let _ = writeln!(md, "- **question:** {}", collapse_ws(question));
    let _ = writeln!(md, "- **requested_at:** {requested_at}\n");

    // Partition the retrieval groups by their target section. Order within each
    // section follows the order the groups were supplied in, which is the order
    // the actions were planned — stable and meaningful for the reader.
    let web = groups_in_section(groups, RetrievalSection::WebSearch);
    let data_lake = groups_in_section(groups, RetrievalSection::DataLake);
    let knowledge = groups_in_section(groups, RetrievalSection::KnowledgeBase);
    let connections = groups_in_section(groups, RetrievalSection::Connections);

    // 1. WebSearchResults
    let _ = writeln!(md, "## WebSearchResults\n");
    render_section(&mut md, &web);

    // 2. DataLakeResults (conversation data lake + diary)
    let _ = writeln!(md, "## DataLakeResults\n");
    render_section(&mut md, &data_lake);

    // 3. KnowledgeBaseResults (retrieved KB records + facts extracted this turn)
    let _ = writeln!(md, "## KnowledgeBaseResults\n");
    render_section(&mut md, &knowledge);
    render_facts(&mut md, &extras.facts);

    // 4. ConnectionsResults
    let _ = writeln!(md, "## ConnectionsResults\n");
    render_section(&mut md, &connections);

    // 5-7. Analysis-derived sections.
    let _ = writeln!(md, "## EmotionResults\n");
    let _ = writeln!(md, "{}\n", value_or_placeholder(&extras.analysis.emotion));
    let _ = writeln!(md, "## TruthfulNessResults\n");
    let _ = writeln!(
        md,
        "{}\n",
        value_or_placeholder(&extras.analysis.truthfulness)
    );
    let _ = writeln!(md, "## IntentionResults\n");
    let _ = writeln!(md, "{}\n", value_or_placeholder(&extras.analysis.intention));

    // 8. OldContextSummary (compressed carry-over from newContext.json)
    let _ = writeln!(md, "## OldContextSummary\n");
    let _ = writeln!(md, "{}", value_or_placeholder(&extras.old_context_summary));

    md
}

/// Decide which of the four retrieval sections a container belongs to.
///
/// The match is keyword-based and case-insensitive, with `DataLake` as a
/// lossless catch-all so an unrecognised container is still surfaced rather
/// than dropped.
fn section_for_container(container: &str) -> RetrievalSection {
    let lowered = container.to_ascii_lowercase();
    if lowered.contains("web") {
        RetrievalSection::WebSearch
    } else if lowered.contains("connection") {
        RetrievalSection::Connections
    } else if lowered.contains("kb") || lowered.contains("knowledge") {
        RetrievalSection::KnowledgeBase
    } else {
        // Conversation data lake, diary, and anything unrecognised land here.
        RetrievalSection::DataLake
    }
}

/// Collect references to the groups that belong to a given section, preserving
/// their original order.
fn groups_in_section(groups: &[RetrievalGroup], section: RetrievalSection) -> Vec<&RetrievalGroup> {
    groups
        .iter()
        .filter(|group| section_for_container(&group.container) == section)
        .collect()
}

/// Render all groups for one section, or a placeholder when there are none.
fn render_section(md: &mut String, groups: &[&RetrievalGroup]) {
    if groups.is_empty() {
        let _ = writeln!(md, "_No results gathered this turn._\n");
        return;
    }
    for group in groups {
        // Per-container sub-heading so each source stays identifiable even when
        // several containers fold into the same section.
        let _ = writeln!(md, "### {} (`{}`)\n", group.container, group.action_id);
        if group.records.is_empty() {
            let _ = writeln!(md, "_No results._\n");
            continue;
        }
        for record in &group.records {
            render_one_record(md, record);
        }
        let _ = writeln!(md);
    }
}

/// Text fields, in priority order, that hold a record's primary payload. The
/// first non-empty one becomes the record's headline; the rest become metadata.
const PRIMARY_TEXT_KEYS: [&str; 7] = [
    "data",
    "notes",
    "description",
    "snippet",
    "text",
    "summary",
    "value",
];

/// Keys never shown as metadata (raw embedding vectors are noise for an LLM).
const SKIP_META_KEYS: [&str; 2] = ["dataVector", "data_vector"];

/// Render a single record as a bullet: a primary text line plus a compact,
/// deterministic metadata line of the remaining scalar fields.
fn render_one_record(md: &mut String, record: &Value) {
    match record {
        Value::Object(map) => {
            // Pick the first non-empty primary text field as the headline.
            let primary = PRIMARY_TEXT_KEYS.iter().find_map(|key| {
                map.get(*key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(|text| (*key, text.to_string()))
            });
            let primary_key = primary.as_ref().map(|(key, _)| *key).unwrap_or("");
            match &primary {
                Some((_, text)) => {
                    let _ = writeln!(md, "- {}", collapse_ws(text));
                }
                None => {
                    let _ = writeln!(md, "- (no text payload)");
                }
            }
            let meta = render_metadata(map, primary_key);
            if !meta.is_empty() {
                let _ = writeln!(md, "  - _meta_: {meta}");
            }
        }
        Value::String(text) => {
            let _ = writeln!(md, "- {}", collapse_ws(text));
        }
        other => {
            // Numbers, bools, arrays: render their JSON form so nothing is lost.
            let _ = writeln!(md, "- {other}");
        }
    }
}

/// Build a deterministic, comma-separated metadata line from a record's scalar
/// fields, skipping the primary-text field and any embedding vectors. Keys are
/// sorted so the output never depends on JSON field ordering.
fn render_metadata(map: &Map<String, Value>, primary_key: &str) -> String {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();

    let mut parts: Vec<String> = Vec::new();
    for key in keys {
        if key == primary_key
            || SKIP_META_KEYS
                .iter()
                .any(|skip| skip.eq_ignore_ascii_case(key))
        {
            continue;
        }
        match &map[key] {
            Value::String(text) if !text.trim().is_empty() => {
                parts.push(format!("{key}={}", collapse_ws(text)));
            }
            Value::Number(number) => parts.push(format!("{key}={number}")),
            Value::Bool(flag) => parts.push(format!("{key}={flag}")),
            Value::Object(inner) => {
                // Flatten one level of nested object (e.g. a `metadata` map) so
                // its contents are not lost.
                let mut inner_keys: Vec<&String> = inner.keys().collect();
                inner_keys.sort();
                for inner_key in inner_keys {
                    if let Some(text) = inner[inner_key].as_str() {
                        if !text.trim().is_empty() {
                            parts.push(format!("{key}.{inner_key}={}", collapse_ws(text)));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    parts.join(", ")
}

/// Render the facts extracted this turn under the knowledge-base section.
fn render_facts(md: &mut String, facts: &[TurnFact]) {
    if facts.is_empty() {
        return;
    }
    let _ = writeln!(md, "### Extracted facts (facts.json)\n");
    for fact in facts {
        // Skip wholly empty entries but keep partially-populated ones.
        if fact.fact.trim().is_empty() && fact.value.trim().is_empty() {
            continue;
        }
        let name = if fact.fact.trim().is_empty() {
            "(unnamed)"
        } else {
            fact.fact.trim()
        };
        let _ = writeln!(md, "- **{}:** {}", name, collapse_ws(&fact.value));
    }
    let _ = writeln!(md);
}

/// Collapse all runs of whitespace (including newlines) into single spaces and
/// trim the ends, keeping rendered bullets compact and single-line.
fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Return the trimmed value, or a stable placeholder when it is empty, so an
/// analysis/context section is never blank.
fn value_or_placeholder(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "_Not assessed this turn._".to_string()
    } else {
        collapse_ws(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collapse_ws_flattens_whitespace() {
        assert_eq!(collapse_ws("  a\n\tb   c "), "a b c");
        assert_eq!(collapse_ws(""), "");
    }

    #[test]
    fn value_or_placeholder_handles_empty() {
        assert_eq!(value_or_placeholder("   "), "_Not assessed this turn._");
        assert_eq!(value_or_placeholder(" calm "), "calm");
    }

    #[test]
    fn section_routing_is_keyword_based() {
        assert_eq!(section_for_container("Web"), RetrievalSection::WebSearch);
        assert_eq!(
            section_for_container("GaiaConnections"),
            RetrievalSection::Connections
        );
        assert_eq!(
            section_for_container("GaiaKB"),
            RetrievalSection::KnowledgeBase
        );
        assert_eq!(
            section_for_container("KnowledgeBase"),
            RetrievalSection::KnowledgeBase
        );
        // Diary and the data lake both fold into DataLake.
        assert_eq!(
            section_for_container("GaiaDiary"),
            RetrievalSection::DataLake
        );
        assert_eq!(
            section_for_container("GaiaDataLake"),
            RetrievalSection::DataLake
        );
        // Unknown containers fall through to the lossless catch-all.
        assert_eq!(
            section_for_container("MysterySource"),
            RetrievalSection::DataLake
        );
    }

    #[test]
    fn parse_call1_extras_reads_all_documents() {
        let reply = r#"
        Here is the plan:
        [
          [ { "id": "q1", "kind": "search", "target": "Web" } ],
          { "emotion": "curious", "truthfulness": "honest", "intention": "learn" },
          [ { "fact": "favourite_colour", "value": "blue" } ],
          { "summary": "User asked about colours before." }
        ]
        "#;

        let extras = parse_call1_extras(reply);
        assert_eq!(extras.analysis.emotion, "curious");
        assert_eq!(extras.analysis.truthfulness, "honest");
        assert_eq!(extras.analysis.intention, "learn");
        assert_eq!(extras.facts.len(), 1);
        assert_eq!(extras.facts[0].fact, "favourite_colour");
        assert_eq!(extras.facts[0].value, "blue");
        assert_eq!(
            extras.old_context_summary,
            "User asked about colours before."
        );
    }

    #[test]
    fn parse_call1_extras_degrades_gracefully() {
        // No array at all → all defaults.
        let extras = parse_call1_extras("no json here");
        assert_eq!(extras, Call1Extras::default());

        // Only the actions element present → analysis/facts/summary stay empty.
        let extras = parse_call1_extras(r#"[ [ { "id": "q1" } ] ]"#);
        assert_eq!(extras.analysis, TurnAnalysis::default());
        assert!(extras.facts.is_empty());
        assert!(extras.old_context_summary.is_empty());
    }

    #[test]
    fn build_emits_all_eight_headings_in_order() {
        let md = build_response_data_context(
            "threadkeeper",
            "What is my favourite colour?",
            "2026-06-16T00:00:00Z",
            &Call1Extras::default(),
            &[],
        );

        let headings = [
            "## WebSearchResults",
            "## DataLakeResults",
            "## KnowledgeBaseResults",
            "## ConnectionsResults",
            "## EmotionResults",
            "## TruthfulNessResults",
            "## IntentionResults",
            "## OldContextSummary",
        ];

        // Each heading must appear, and in strictly increasing position.
        let mut last = 0;
        for heading in headings {
            let at = md
                .find(heading)
                .unwrap_or_else(|| panic!("missing heading {heading}"));
            assert!(at >= last, "heading out of order: {heading}");
            last = at;
        }

        // Empty inputs still produce stable placeholders.
        assert!(md.contains("_No results gathered this turn._"));
        assert!(md.contains("_Not assessed this turn._"));
    }

    #[test]
    fn build_renders_records_into_correct_sections() {
        let groups = vec![
            RetrievalGroup {
                action_id: "q1".to_string(),
                container: "Web".to_string(),
                records: vec![json!({
                    "title": "Colour theory",
                    "url": "https://example.com/colour",
                    "snippet": "An overview of colours."
                })],
            },
            RetrievalGroup {
                action_id: "q3".to_string(),
                container: "GaiaKB".to_string(),
                records: vec![json!({
                    "id": "GaiaKB|threadkeeper|2026-06-16",
                    "entity": "threadkeeper",
                    "date": "2026-06-16",
                    "data": "User's favourite colour is blue.",
                    "dataVector": [0.1, 0.2, 0.3]
                })],
            },
            RetrievalGroup {
                action_id: "q5".to_string(),
                container: "GaiaConnections".to_string(),
                records: vec![json!({
                    "entity": "threadkeeper",
                    "notes": "Shared a warm moment.",
                    "changeAmount": 5
                })],
            },
        ];

        let extras = Call1Extras {
            analysis: TurnAnalysis {
                emotion: "warm".to_string(),
                truthfulness: "honest".to_string(),
                intention: "reconnect".to_string(),
            },
            facts: vec![TurnFact {
                fact: "favourite_colour".to_string(),
                value: "blue".to_string(),
            }],
            old_context_summary: "Earlier they discussed colours.".to_string(),
        };

        let md = build_response_data_context(
            "threadkeeper",
            "How are we doing?",
            "2026-06-16T00:00:00Z",
            &extras,
            &groups,
        );

        // Web record under WebSearchResults; the embedding vector is omitted.
        assert!(md.contains("An overview of colours."));
        assert!(md.contains("url=https://example.com/colour"));
        assert!(!md.contains("dataVector"));

        // KB record headline plus extracted facts both appear in the KB section.
        assert!(md.contains("User's favourite colour is blue."));
        assert!(md.contains("### Extracted facts (facts.json)"));
        assert!(md.contains("**favourite_colour:** blue"));

        // Connections record uses `notes` as its primary text.
        assert!(md.contains("Shared a warm moment."));

        // Analysis values land in their sections.
        assert!(md.contains("warm"));
        assert!(md.contains("reconnect"));
        assert!(md.contains("Earlier they discussed colours."));
    }
}
