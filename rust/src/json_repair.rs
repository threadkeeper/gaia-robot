//! Best-effort structural repair for the malformed JSON that LLMs occasionally
//! emit.
//!
//! When asked to return a JSON array of documents, a language model sometimes
//! drops a single structural character at an "exit point" — most often the
//! opening `{` of an array (or object-value) element, e.g. emitting
//! `[ … }, "id": … ]` instead of `[ … }, { "id": … ]`. One dropped brace makes
//! the whole turn unparseable even though the content is otherwise fine, and we
//! have observed it landing on *different* elements from run to run. Rather than
//! patch one position, this module repairs the whole family of common,
//! safe-to-correct structural defects generically.
//!
//! [`repair_json`] performs ONE conservative left-to-right pass that fixes:
//!
//! * a missing object-opening `{` wherever a key appears in a value position
//!   (any array element or object value, at any nesting depth);
//! * trailing commas immediately before `}` or `]`;
//! * a leading `+` on a number value (e.g. `"delta":+2`), which JSON forbids
//!   even though many tolerant parsers accept it;
//! * an unterminated trailing string and any unclosed `{`/`[` at end of input
//!   (truncated output).
//!
//! It is a **no-op on already-valid JSON**: well-formed input is copied through
//! unchanged, because none of the defects above can occur in valid JSON. Callers
//! therefore use it purely as a fallback *after* a strict parse fails, so a
//! correct reply is never altered.

/// One container on the structural stack we walk as we scan the input.
///
/// We only need enough state to answer a single question when we meet a string:
/// "is a *value* expected at this position?" If a string that is actually a key
/// (it is followed by `:`) shows up where a value belongs, the object's opening
/// `{` was dropped and we re-insert it.
#[derive(Clone, Copy)]
struct Frame {
    /// `true` for an array `[ … ]`, `false` for an object `{ … }`.
    is_array: bool,
    /// For objects only: `true` when the next string should be a key, `false`
    /// when a value is expected (i.e. just after a `:`). Unused for arrays,
    /// whose elements are always values.
    expect_key: bool,
}

/// Repair the common structural defects in `input` and return corrected JSON.
///
/// Valid JSON is returned unchanged. See the module docs for the exact set of
/// defects handled. This never fails: in the worst case it returns its best
/// effort, which the caller then tries to parse (and ignores if still invalid).
///
/// # Examples
///
/// ```
/// # // (Exercised via this module's unit tests; the crate is a binary, so the
/// # // function is not part of a public library API.)
/// ```
pub fn repair_json(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len() + 8);
    let mut stack: Vec<Frame> = Vec::new();

    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'"' => {
                // Copy the whole string verbatim, then decide whether it was a
                // key that landed in a value position (a dropped `{`).
                let (end, terminated) = scan_string_end(bytes, i);
                let literal = &input[i..end];

                // A string is a key when the next significant byte is `:`.
                let is_key = terminated && next_significant(bytes, end) == Some(b':');
                // A value is expected at the top level, in any array element, or
                // in an object right after a `:` (expect_key == false).
                let value_expected = match stack.last() {
                    None => true,
                    Some(frame) if frame.is_array => true,
                    Some(frame) => !frame.expect_key,
                };

                if is_key && value_expected {
                    // The opening brace of this object was dropped. Re-create it
                    // and enter a fresh object whose first member is this key.
                    out.push('{');
                    stack.push(Frame {
                        is_array: false,
                        expect_key: true,
                    });
                }

                // Emitting a key satisfies an object's "expect a key" state; the
                // following `:` (handled below) then switches us to value mode.
                if let Some(top) = stack.last_mut() {
                    if !top.is_array && is_key {
                        top.expect_key = false;
                    }
                }

                out.push_str(literal);
                if !terminated {
                    // Truncated mid-string: close it so the tail can parse.
                    out.push('"');
                }
                i = end;
            }
            b'{' => {
                out.push('{');
                stack.push(Frame {
                    is_array: false,
                    expect_key: true,
                });
                i += 1;
            }
            b'[' => {
                out.push('[');
                stack.push(Frame {
                    is_array: true,
                    expect_key: false,
                });
                i += 1;
            }
            b'}' | b']' => {
                out.push(char::from(b));
                stack.pop();
                // The closed container is a completed value in its parent; an
                // object parent now expects the next key (or to close).
                if let Some(top) = stack.last_mut() {
                    if !top.is_array {
                        top.expect_key = true;
                    }
                }
                i += 1;
            }
            b':' => {
                out.push(':');
                if let Some(top) = stack.last_mut() {
                    if !top.is_array {
                        top.expect_key = false; // a value follows the colon
                    }
                }
                i += 1;
            }
            b',' => {
                // Drop a trailing comma (one sitting just before `}` or `]`).
                if matches!(next_significant(bytes, i + 1), Some(b'}') | Some(b']')) {
                    i += 1;
                    continue;
                }
                out.push(',');
                if let Some(top) = stack.last_mut() {
                    if !top.is_array {
                        top.expect_key = true; // the next object token is a key
                    }
                }
                i += 1;
            }
            b'+' => {
                // A leading `+` on a number value is invalid JSON (only `-` is
                // allowed). Drop it when it sits at the start of a value — right
                // after `:`, `,`, or `[` and before a digit. An exponent sign
                // (as in `1e+5`) is preceded by `e`/`E`, never by those, so it
                // is left untouched.
                let starts_value = matches!(
                    prev_significant(bytes, i),
                    Some(b':') | Some(b',') | Some(b'[')
                );
                let digit_follows = bytes.get(i + 1).is_some_and(u8::is_ascii_digit);
                if starts_value && digit_follows {
                    i += 1; // skip the stray '+'
                    continue;
                }
                out.push('+');
                i += 1;
            }
            _ => {
                // Whitespace, numbers, and literal tokens (true/false/null) are
                // all ASCII outside strings and copy through unchanged.
                out.push(char::from(b));
                i += 1;
            }
        }
    }

    // Close anything the model left open (truncated output), innermost first.
    for frame in stack.iter().rev() {
        out.push(if frame.is_array { ']' } else { '}' });
    }

    out
}

/// Find the byte index just past the closing quote of the string that starts at
/// `start` (where `bytes[start]` is `"`).
///
/// Returns `(end, terminated)`: `end` is the index after the closing quote (or
/// `bytes.len()` if the string runs off the end), and `terminated` is `true`
/// only when a real closing quote was found. JSON escape sequences are skipped
/// so an escaped quote does not end the string early.
fn scan_string_end(bytes: &[u8], start: usize) -> (usize, bool) {
    let mut j = start + 1;
    while j < bytes.len() {
        match bytes[j] {
            b'\\' => j += 2, // skip the escaped character
            b'"' => return (j + 1, true),
            _ => j += 1,
        }
    }
    (bytes.len(), false)
}

/// Return the first non-whitespace byte at or after `from`, or `None` at end of
/// input. Used for the small look-aheads that classify keys and trailing commas.
fn next_significant(bytes: &[u8], from: usize) -> Option<u8> {
    let mut j = from;
    while j < bytes.len() {
        if !bytes[j].is_ascii_whitespace() {
            return Some(bytes[j]);
        }
        j += 1;
    }
    None
}

/// Return the first non-whitespace byte strictly *before* `before`, or `None` at
/// the start of input. Used to tell a number's leading sign (in value position)
/// from an exponent sign.
fn prev_significant(bytes: &[u8], before: usize) -> Option<u8> {
    let mut j = before;
    while j > 0 {
        j -= 1;
        if !bytes[j].is_ascii_whitespace() {
            return Some(bytes[j]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse with serde to prove the repaired text is genuinely valid JSON and
    /// equals the expected value.
    fn assert_repairs_to(input: &str, expected_json: &str) {
        let repaired = repair_json(input);
        let got: serde_json::Value = serde_json::from_str(&repaired)
            .unwrap_or_else(|e| panic!("not valid JSON: {e}\n{repaired}"));
        let want: serde_json::Value = serde_json::from_str(expected_json).unwrap();
        assert_eq!(got, want, "repaired: {repaired}");
    }

    #[test]
    fn leaves_valid_json_unchanged() {
        for valid in [
            r#"[]"#,
            r#"{}"#,
            r#"[1,2,3]"#,
            r#"["id","kind"]"#,
            r#"[{"a":1},{"b":2}]"#,
            r#"{"k":{"nested":[1,{"x":true}]},"s":"a:b, c"}"#,
            r#"[{"text":"he said \"hi\":","n":-0.33,"ok":true,"z":null}]"#,
        ] {
            assert_eq!(repair_json(valid), valid, "should be a no-op");
        }
    }

    #[test]
    fn restores_a_dropped_object_brace_in_an_array() {
        // `},"id"` should become `},{"id"` — the exact bug seen in the probe.
        let input = r#"[{"id":"a1"},"id":"a2"}]"#;
        assert_repairs_to(input, r#"[{"id":"a1"},{"id":"a2"}]"#);
    }

    #[test]
    fn restores_a_dropped_brace_for_the_first_array_element() {
        let input = r#"["id":"a1","k":1}]"#;
        assert_repairs_to(input, r#"[{"id":"a1","k":1}]"#);
    }

    #[test]
    fn restores_a_dropped_brace_in_object_value_position() {
        // `"payload":"entity":` — a key where the value of `payload` belongs.
        let input = r#"{"payload":"entity":"x"}}"#;
        assert_repairs_to(input, r#"{"payload":{"entity":"x"}}"#);
    }

    #[test]
    fn restores_multiple_dropped_braces_at_different_positions() {
        // Mirrors the real failures: braces dropped before two later elements.
        let input = r#"[{"id":"a1","r":"x"},"id":"a2","r":"y"},"id":"a3","r":"z"}]"#;
        assert_repairs_to(
            input,
            r#"[{"id":"a1","r":"x"},{"id":"a2","r":"y"},{"id":"a3","r":"z"}]"#,
        );
    }

    #[test]
    fn removes_trailing_commas() {
        assert_repairs_to(r#"[1,2,3,]"#, r#"[1,2,3]"#);
        assert_repairs_to(r#"{"a":1,"b":2,}"#, r#"{"a":1,"b":2}"#);
        assert_repairs_to(r#"[{"a":1,},]"#, r#"[{"a":1}]"#);
    }

    #[test]
    fn strips_a_leading_plus_on_number_values() {
        // The exact bug seen in the probe: `"delta":+2`.
        assert_repairs_to(r#"{"delta":+2}"#, r#"{"delta":2}"#);
        assert_repairs_to(r#"[+1,+2,+3]"#, r#"[1,2,3]"#);
        // An exponent sign must be preserved (it is valid JSON).
        assert_repairs_to(r#"{"n":1e+5}"#, r#"{"n":1e+5}"#);
        // A '+' inside a string must be left alone.
        assert_repairs_to(r#"{"s":"+2 points"}"#, r#"{"s":"+2 points"}"#);
    }

    #[test]
    fn closes_unclosed_containers_at_eof() {
        assert_repairs_to(r#"[{"a":1},{"b":2"#, r#"[{"a":1},{"b":2}]"#);
        assert_repairs_to(r#"{"a":[1,2"#, r#"{"a":[1,2]}"#);
    }

    #[test]
    fn closes_an_unterminated_trailing_string() {
        assert_repairs_to(r#"[{"text":"cut off here"#, r#"[{"text":"cut off here"}]"#);
    }

    #[test]
    fn brackets_inside_strings_do_not_confuse_the_scanner() {
        // The `}`/`{` inside the string must not affect the structural stack.
        let input = r#"[{"note":"a } b { c"},"id":"a2"}]"#;
        assert_repairs_to(input, r#"[{"note":"a } b { c"},{"id":"a2"}]"#);
    }
}
