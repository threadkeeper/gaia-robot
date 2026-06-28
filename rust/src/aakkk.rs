//! Coconut step 5 — the **AAKKK** compression format.
//!
//! AAKKK ("Atomic Attribute / Key-Key-Key") is the dense, canonical one-line
//! form Coconut packs into the context window. Where `raw` text is kept for
//! fidelity, the AAKKK line is what actually gets counted and packed at query
//! time, so it must be:
//!
//! - **dense** — fluff stripped, whitespace collapsed (typically 50–70% fewer
//!   tokens than the raw text);
//! - **canonical** — the same inputs always render the exact same line, so token
//!   counts and packing are stable and debuggable;
//! - **machine-parseable** — fixed delimiters with escaping so a rendered line
//!   can be split back into its fields.
//!
//! A line looks like:
//!
//! ```text
//! A:entity=Jonty|A:action=keen_on|A:topic=Coconut|K:salience=0.95|K:tokens=42|K:ts=20260628T0458
//! ```
//!
//! `A:` fields are *atomic attributes* of the memory (entity, action, topic,
//! a compressed text summary); `K:` fields are *keys/metadata* (salience, token
//! count, timestamp). Values are escaped so the `|` separator and `=`
//! key/value delimiter can appear safely inside a value.

// The Coconut surface is being scaffolded incrementally; not every public item
// is wired into the engine yet. Mirrors `crate::storage` / `write_data_controller`.
#![allow(dead_code)]

/// The reserved characters that structure an AAKKK line. A value containing any
/// of these (or a backslash) is escaped on render and unescaped on parse.
const ESCAPE_PREFIX: char = '\\';

/// A small, fixed set of English filler words removed by [`strip_fluff`]. The
/// list is deliberately conservative: it drops only low-signal connective words
/// so the compressed text stays readable and the transform stays predictable.
const FILLER_WORDS: [&str; 12] = [
    "the", "a", "an", "is", "are", "of", "to", "and", "that", "this", "it", "in",
];

/// One canonical AAKKK line: an ordered set of `A:` atomic attributes followed
/// by an ordered set of `K:` metadata keys.
///
/// Build a line with [`AakkkLine::new`] and the [`AakkkLine::attr`] /
/// [`AakkkLine::key`] builders (insertion order is preserved so rendering is
/// deterministic), then [`AakkkLine::render`] it. A rendered line can be read
/// back with [`AakkkLine::parse`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AakkkLine {
    /// `A:` atomic attributes, in insertion order.
    attributes: Vec<(String, String)>,
    /// `K:` metadata keys, in insertion order.
    keys: Vec<(String, String)>,
}

impl AakkkLine {
    /// Create an empty line.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an `A:` atomic attribute (e.g. `entity`, `action`, `topic`).
    ///
    /// Empty values are skipped so absent fields never clutter the line — a
    /// record with no `action` simply omits `A:action=` rather than emitting an
    /// empty one.
    pub fn attr(mut self, name: &str, value: &str) -> Self {
        if !value.is_empty() {
            self.attributes.push((name.to_string(), value.to_string()));
        }
        self
    }

    /// Append a `K:` metadata key (e.g. `salience`, `tokens`, `ts`). Empty
    /// values are skipped, mirroring [`AakkkLine::attr`].
    pub fn key(mut self, name: &str, value: &str) -> Self {
        if !value.is_empty() {
            self.keys.push((name.to_string(), value.to_string()));
        }
        self
    }

    /// Render the canonical single-line AAKKK string.
    ///
    /// Fields are emitted in insertion order, `A:` attributes first then `K:`
    /// keys, joined by `|`. Each value is escaped so the structural characters
    /// can appear inside it. The output never contains a newline.
    pub fn render(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(self.attributes.len() + self.keys.len());
        for (name, value) in &self.attributes {
            parts.push(format!("A:{}={}", name, escape(value)));
        }
        for (name, value) in &self.keys {
            parts.push(format!("K:{}={}", name, escape(value)));
        }
        parts.join("|")
    }

    /// Parse a rendered AAKKK line back into its fields.
    ///
    /// This is the inverse of [`AakkkLine::render`]: it splits on unescaped `|`,
    /// classifies each field by its `A:`/`K:` prefix, and unescapes the value.
    /// Fields without a recognised prefix or without an `=` are skipped, so the
    /// parser never fails on slightly malformed input.
    pub fn parse(line: &str) -> Self {
        let mut out = Self::new();
        for field in split_unescaped(line, '|') {
            let (prefix, rest) = match field.split_once(':') {
                Some(pair) => pair,
                None => continue,
            };
            let (name, value) = match rest.split_once('=') {
                Some(pair) => pair,
                None => continue,
            };
            let value = unescape(value);
            match prefix {
                "A" => out.attributes.push((name.to_string(), value)),
                "K" => out.keys.push((name.to_string(), value)),
                _ => {}
            }
        }
        out
    }

    /// The `A:` atomic attributes, in order.
    pub fn attributes(&self) -> &[(String, String)] {
        &self.attributes
    }

    /// The `K:` metadata keys, in order.
    pub fn keys(&self) -> &[(String, String)] {
        &self.keys
    }
}

/// Escape the AAKKK-structural characters (`\`, `|`, `=`) and collapse any
/// newlines to spaces so a value can be embedded in a line safely.
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' | '|' | '=' => {
                out.push(ESCAPE_PREFIX);
                out.push(ch);
            }
            '\n' | '\r' => out.push(' '),
            other => out.push(other),
        }
    }
    out
}

/// Reverse [`escape`]: turn `\x` escape pairs back into the literal character.
fn unescape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch == ESCAPE_PREFIX {
            // A trailing lone backslash is kept verbatim rather than dropped.
            if let Some(escaped) = chars.next() {
                out.push(escaped);
            } else {
                out.push(ESCAPE_PREFIX);
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Split `line` on `sep`, but ignore separators that are preceded by the escape
/// prefix (so an escaped `|` inside a value does not start a new field).
fn split_unescaped(line: &str, sep: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for ch in line.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
        } else if ch == ESCAPE_PREFIX {
            current.push(ch);
            escaped = true;
        } else if ch == sep {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

/// Compress free text into a dense, lower-token form suitable for an AAKKK `A:`
/// value: collapse all whitespace runs to single spaces and drop a small set of
/// low-signal [`FILLER_WORDS`] (case-insensitively), preserving word order.
///
/// This is a deliberately simple, deterministic first pass (it can be made
/// smarter later without changing the format). The high-fidelity original is
/// always kept separately as the record's `raw`/`data`, so this lossy
/// compression only ever affects packing density, never stored fidelity.
pub fn strip_fluff(text: &str) -> String {
    let kept: Vec<&str> = text
        .split_whitespace()
        .filter(|word| {
            let lowered = word.to_ascii_lowercase();
            // Strip surrounding punctuation before testing so "the," still counts
            // as the filler word "the".
            let trimmed = lowered.trim_matches(|c: char| !c.is_alphanumeric());
            !FILLER_WORDS.contains(&trimmed)
        })
        .collect();
    kept.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_emits_attributes_then_keys_in_order() {
        let line = AakkkLine::new()
            .attr("entity", "Jonty")
            .attr("action", "keen_on")
            .attr("topic", "Coconut")
            .key("salience", "0.95")
            .key("tokens", "42")
            .key("ts", "20260628T0458");

        assert_eq!(
            line.render(),
            "A:entity=Jonty|A:action=keen_on|A:topic=Coconut|K:salience=0.95|K:tokens=42|K:ts=20260628T0458"
        );
    }

    #[test]
    fn empty_values_are_omitted() {
        let line = AakkkLine::new()
            .attr("entity", "rust")
            .attr("action", "")
            .key("tokens", "7")
            .key("ts", "");
        assert_eq!(line.render(), "A:entity=rust|K:tokens=7");
    }

    #[test]
    fn values_with_structural_characters_round_trip() {
        let line = AakkkLine::new()
            .attr("text", "a|b=c\\d")
            .key("note", "x=y|z");
        let rendered = line.render();
        // The structural characters are escaped in the rendered form.
        assert!(rendered.contains("a\\|b\\=c\\\\d"));
        // ...and parse fully recovers the original values.
        let parsed = AakkkLine::parse(&rendered);
        assert_eq!(parsed, line);
    }

    #[test]
    fn parse_skips_malformed_fields() {
        let parsed = AakkkLine::parse("A:entity=rust|garbage|K:tokens=3|X:other=z");
        assert_eq!(parsed.attributes(), &[("entity".into(), "rust".into())]);
        assert_eq!(parsed.keys(), &[("tokens".into(), "3".into())]);
    }

    #[test]
    fn render_never_contains_a_newline() {
        let line = AakkkLine::new().attr("text", "line one\nline two\r\nthree");
        assert!(!line.render().contains('\n'));
        assert!(!line.render().contains('\r'));
    }

    #[test]
    fn strip_fluff_collapses_whitespace_and_drops_filler() {
        let compressed = strip_fluff("The  quick brown fox\n is a master of the hunt");
        assert_eq!(compressed, "quick brown fox master hunt");
    }

    #[test]
    fn strip_fluff_is_deterministic() {
        let input = "this   is\tThe   example of A test";
        assert_eq!(strip_fluff(input), strip_fluff(input));
        assert_eq!(strip_fluff(input), "example test");
    }
}
