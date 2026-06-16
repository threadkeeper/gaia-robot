//! The [`Call1Prompt`] type: the exact message pair sent to LLM Call 1.
//!
//! LLM Call 1 is the **pull / research** pass. Its job is *not* to answer the
//! user yet, but to look at the user's sentence plus Gaia's running context and
//! decide *what to go and fetch*. It does that by emitting four JSON documents
//! (`actions.json`, `analysis.json`, `facts.json`, `newContext.json`) as a
//! single JSON array.
//!
//! The prompt is formed in this shape (dictated by the design):
//!
//! > You are Gaia, the legendary robot from the Asimov novels, and are chatting
//! > to the human `<user_id>`. He has sent you the following input `<input>`,
//! > and you currently have the following context `<conversation history>`. In
//! > order to answer you must first do research by compiling documents that will
//! > be used for searching in your tailor-made tools. Document spec: `<...>` Tool
//! > spec: `<...>` Only output a single JSON array of the 4 JSON documents.
//!
//! We map that onto the chat API as two messages: the **system** message holds
//! the stable framing (identity, the document/tool specs, and the
//! output-format rule), and the **user** message holds the per-turn data (the
//! human's input, the current time, and the conversation history). Keeping the
//! heavy, unchanging specs in the system message means they stay identical turn
//! to turn while only the small user message varies.

use std::time::{SystemTime, UNIX_EPOCH};

/// Data contracts for the four JSON documents LLM Call 1 must emit, in order.
///
/// These mirror the real Rust contracts: `actions.json` matches
/// [`crate::actions::ActionsFile`], and the remaining three match the document
/// shapes described by the flow diagram. The model is told to emit all four as a
/// single JSON array.
const DOCUMENT_SPEC: &str = "\
1. actions.json - READ-ONLY research queries only (no side effects). Shape:
   { \"version\": \"1.0\",
     \"session\": { \"user_id\": \"<this user>\", \"requested_at\": \"<use the current time given in the user message>\" },
     \"actions\": [
       { \"id\": \"q1\", \"kind\": \"query\",
         \"target\": \"UsersKB|UsersDL|GaiaKB|GaiaLH|GaiaCosmos|GaiaConnections\",
         \"user_id\": \"<required for Users* targets, otherwise null>\",
         \"entity\": \"<subject to search for, optional>\",
         \"intent\": \"<natural-language description of what to retrieve>\",
         \"top\": 3,
         \"filters\": { \"from_date\": null, \"to_date\": null, \"text\": null, \"semantic\": null } }
     ] }
2. analysis.json - your read of the user this turn. Shape:
   { \"emotion\": \"<value>\", \"truthfulness\": \"<value>\", \"intention\": \"<value>\" }
3. facts.json - durable facts worth remembering about the user/world. Shape:
   [ { \"fact\": \"<short key>\", \"value\": \"<value>\" } ]
4. newContext.json - a compressed (~61%) carry-over summary of this turn's
   context plus your reasoning, preserving WHAT you decided to search for and
   WHY. Shape:
   { \"summary\": \"<compressed context>\" }";

/// The retrieval tools Call 1 may target through `actions.json`.
///
/// Each line names a target container (or the web), how it is partitioned, and
/// whether a `user_id` is mandatory. `Users*` targets are per-user and must be
/// scoped to the current user only, which is how user isolation is enforced.
const TOOL_SPEC: &str = "\
- UsersDL   (semantic): this user's data lake; partition=userId; user_id REQUIRED.
- UsersKB   (semantic): this user's knowledge base; partition=userId; user_id REQUIRED.
- GaiaKB    (semantic): Gaia's shared knowledge base; partition=entity.
- GaiaLH    (logical) : Gaia's logical history; partition=entity.
- GaiaCosmos (named)  : Gaia's cosmos index; partition=entity.
- GaiaConnections     : per-user emotional-bank-account ledger; partition=entity.
- Web       (search)  : public web search; results are logged to the Search History.
Every query defaults to top=3. Users* targets MUST set user_id to this user only.";

/// The fully-formed prompt for LLM Call 1, split into the two chat messages.
///
/// Build one with [`Call1Prompt::build`]. The `system` message carries Gaia's
/// identity plus the document and tool specs; the `user` message carries the
/// human's input and the current conversation history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call1Prompt {
    /// The system message: identity, document spec, tool spec, output rule.
    pub system: String,
    /// The user message: this turn's input plus the conversation history.
    pub user: String,
}

impl Call1Prompt {
    /// Form the Call 1 prompt for one turn.
    ///
    /// `user_id` is the human Gaia is chatting to (used for both the greeting and
    /// user isolation). `input` is the human's raw sentence. `conversation_history`
    /// is Gaia's current running context; when empty, a clear placeholder is used
    /// so the model is never handed a dangling "context:" with nothing after it.
    /// `requested_at` is the current time (see [`now_rfc3339`]); the model is told
    /// to reuse it for the `requested_at` field rather than inventing one.
    pub fn build(
        user_id: &str,
        input: &str,
        conversation_history: &str,
        requested_at: &str,
    ) -> Self {
        // Never hand the model an empty context tail; say so explicitly instead.
        let history = if conversation_history.trim().is_empty() {
            "(no prior conversation yet)"
        } else {
            conversation_history
        };

        // System message: the stable framing that does not change per turn.
        let system = format!(
            "You are Gaia, the legendary robot from the Asimov novels, and you are \
             chatting to the human \"{user_id}\". In order to answer you must first \
             do research by compiling documents that will be used for searching in \
             your tailor-made tools.\n\n\
             Document spec:\n{DOCUMENT_SPEC}\n\n\
             Tool spec:\n{TOOL_SPEC}\n\n\
             Only output a single JSON array containing the 4 JSON documents in this \
             order: actions.json, analysis.json, facts.json, newContext.json. Output \
             nothing else - no prose and no markdown code fences."
        );

        // User message: the small, per-turn payload, stamped with the real time.
        let user = format!(
            "The current time is {requested_at} (use this exact value for \
             requested_at).\n\n\
             The human \"{user_id}\" has sent you the following input:\n{input}\n\n\
             You currently have the following context (conversation history):\n{history}"
        );

        Self { system, user }
    }
}

/// Data contracts for the documents LLM Call 2 must emit, in order.
///
/// Call 2 is the **push / answer** pass. It has already been handed the
/// Response Data Context (the assembled research results from Call 1), so its
/// job is to (1) write Gaia's reply and (2) plan any side effects. It emits two
/// documents as a single JSON array: `response.json` then `actions.json`.
const CALL2_DOCUMENT_SPEC: &str = "\
1. response.json - Gaia's actual reply to the human, in her voice. Shape:
   { \"text\": \"<the reply to show the user>\",
     \"emote\": \"<optional one-word emotional cue, e.g. warm|playful|concerned>\",
     \"medium\": \"console|whatsapp|push\" }
2. actions.json - WRITE / side-effecting actions to carry out after replying
   (POST). Every entry is an effect, never a read. Shape:
   { \"version\": \"1.0\",
     \"session\": { \"user_id\": \"<this user>\", \"requested_at\": \"<use the current time given in the user message>\" },
     \"actions\": [
       { \"id\": \"a1\",
         \"kind\": \"upsert|send|actuate|connection\",
         \"target\": \"UsersKB|UsersDL|GaiaKB|GaiaLH|GaiaCosmos|WhatsApp|Push|Actuator|GaiaConnections\",
         \"user_id\": \"<required for Users* targets, otherwise null>\",
         \"payload\": { \"<effect-specific fields>\": \"<value>\" },
         \"reason\": \"<why this side effect is justified by the context>\" }
     ] }";

/// The side-effecting tools/effects Call 2 may target through `actions.json`.
///
/// Unlike Call 1's read-only retrieval tools, every effect here changes state:
/// it writes a memory, sends a message, moves an actuator, or adjusts the
/// friendship ledger. `Users*` write-backs must stay scoped to the current user.
const CALL2_ACTION_SPEC: &str = "\
- upsert  -> UsersKB|UsersDL|GaiaKB|GaiaLH|GaiaCosmos : write/update a memory
             record. payload = { \"id\": \"<optional>\", \"entity\": \"<subject>\",
             \"data\": \"<text>\" }. Users* writes REQUIRE user_id (this user only).
- send    -> WhatsApp|Push : deliver a message to the user. payload =
             { \"text\": \"<message>\" }.
- actuate -> Actuator : drive a physical/robot output. payload =
             { \"device\": \"<name>\", \"command\": \"<value>\" }.
- connection -> GaiaConnections : adjust the emotional-bank-account balance for
             this user. payload = { \"delta\": <signed integer>, \"note\":
             \"<why the balance changed this turn>\" }.
Only emit effects that are clearly justified by the Response Data Context. If no
side effect is warranted, return an empty actions array.";

/// The fully-formed prompt for LLM Call 2, split into the two chat messages.
///
/// Build one with [`Call2Prompt::build`]. The `system` message carries Gaia's
/// identity plus the document and action specs; the `user` message carries the
/// human's input, the current time, and the assembled Response Data Context
/// produced by executing Call 1's plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call2Prompt {
    /// The system message: identity, document spec, action spec, output rule.
    pub system: String,
    /// The user message: this turn's input, the time, and the research results.
    pub user: String,
}

impl Call2Prompt {
    /// Form the Call 2 prompt for one turn.
    ///
    /// `user_id` is the human Gaia is chatting to (greeting + user isolation).
    /// `input` is the human's original sentence. `response_data_context` is the
    /// assembled research output from Call 1 (see
    /// `research/ResponseDataContext.md`); when empty, a clear placeholder is
    /// used so the model is never handed a dangling "context:" with nothing
    /// after it. `requested_at` is the current time (see [`now_rfc3339`]); the
    /// model is told to reuse it for the `requested_at` field.
    pub fn build(
        user_id: &str,
        input: &str,
        response_data_context: &str,
        requested_at: &str,
    ) -> Self {
        // Never hand the model an empty context tail; say so explicitly instead.
        let context = if response_data_context.trim().is_empty() {
            "(no research results were assembled this turn)"
        } else {
            response_data_context
        };

        // System message: the stable framing that does not change per turn.
        let system = format!(
            "You are Gaia, the legendary robot from the Asimov novels, and you are \
             chatting to the human \"{user_id}\". You have already done your research \
             and been handed a Response Data Context containing the results. Now \
             answer the human and decide what side effects to carry out.\n\n\
             Document spec:\n{CALL2_DOCUMENT_SPEC}\n\n\
             Action spec:\n{CALL2_ACTION_SPEC}\n\n\
             Ground every claim in the Response Data Context - never invent facts \
             that are not present there. Only output a single JSON array containing \
             the 2 JSON documents in this order: response.json, actions.json. Output \
             nothing else - no prose and no markdown code fences."
        );

        // User message: the per-turn payload plus the assembled research results.
        let user = format!(
            "The current time is {requested_at} (use this exact value for \
             requested_at).\n\n\
             The human \"{user_id}\" originally sent you the following input:\n{input}\n\n\
             Here is your Response Data Context (the research results to ground your \
             answer in):\n{context}"
        );

        Self { system, user }
    }
}

/// `2026-06-16T12:00:00Z`.
///
/// Implemented with the standard library only (no `chrono`/`time` dependency),
/// reusing Howard Hinnant's civil-from-days date algorithm. Used to stamp the
/// Call 1 prompt with a real time so the model does not guess `requested_at`.
pub fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    rfc3339(secs)
}

/// Format seconds-since-Unix-epoch as `2026-06-16T12:00:00Z` (UTC).
fn rfc3339(secs_since_epoch: u64) -> String {
    let days = (secs_since_epoch / 86_400) as i64;
    let secs_of_day = secs_since_epoch % 86_400;
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Convert a day count since 1970-01-01 into a `(year, month, day)` triple.
///
/// Howard Hinnant's well-known `civil_from_days` algorithm; the magic constants
/// come from the proleptic Gregorian calendar's 400-year cycle.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year, [0, 365]
    let mp = (5 * doy + 2) / 153; // month index, [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_embeds_user_input_and_history() {
        let prompt = Call1Prompt::build(
            "threadkeeper",
            "what do you know about me?",
            "we met once",
            "2026-06-16T12:00:00Z",
        );

        // The user id appears in both the greeting and the user message.
        assert!(prompt.system.contains("\"threadkeeper\""));
        assert!(prompt.user.contains("\"threadkeeper\""));
        // The raw input and the supplied history are threaded into the user message.
        assert!(prompt.user.contains("what do you know about me?"));
        assert!(prompt.user.contains("we met once"));
        // The supplied timestamp is injected for the model to reuse.
        assert!(prompt.user.contains("2026-06-16T12:00:00Z"));
    }

    #[test]
    fn empty_history_becomes_a_placeholder() {
        let prompt = Call1Prompt::build("threadkeeper", "hi", "   ", "2026-06-16T12:00:00Z");
        assert!(prompt.user.contains("(no prior conversation yet)"));
    }

    #[test]
    fn system_lists_all_four_documents_and_the_output_rule() {
        let prompt = Call1Prompt::build("threadkeeper", "hi", "", "2026-06-16T12:00:00Z");

        for document in [
            "actions.json",
            "analysis.json",
            "facts.json",
            "newContext.json",
        ] {
            assert!(prompt.system.contains(document), "missing {document}");
        }
        // The single-array output rule must be present and unambiguous.
        assert!(prompt.system.contains("single JSON array"));
    }

    #[test]
    fn system_lists_every_retrieval_tool() {
        let prompt = Call1Prompt::build("threadkeeper", "hi", "", "2026-06-16T12:00:00Z");

        for target in [
            "UsersDL",
            "UsersKB",
            "GaiaKB",
            "GaiaLH",
            "GaiaCosmos",
            "GaiaConnections",
            "Web",
        ] {
            assert!(prompt.system.contains(target), "missing {target}");
        }
    }

    #[test]
    fn rfc3339_formats_the_unix_epoch() {
        // Day 0 of the Unix epoch is 1970-01-01T00:00:00Z.
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        // 2026-06-16T12:00:00Z is 20620 whole days plus 12 hours after the epoch.
        let secs = 20_620u64 * 86_400 + 43_200;
        assert_eq!(rfc3339(secs), "2026-06-16T12:00:00Z");
    }

    #[test]
    fn call2_embeds_input_context_and_timestamp() {
        let prompt = Call2Prompt::build(
            "threadkeeper",
            "what do you know about me?",
            "retrievals: UsersKB -> prefers concise answers",
            "2026-06-16T12:00:00Z",
        );

        // The user id appears in both the greeting and the user message.
        assert!(prompt.system.contains("\"threadkeeper\""));
        assert!(prompt.user.contains("\"threadkeeper\""));
        // The original input and the supplied research context are threaded in.
        assert!(prompt.user.contains("what do you know about me?"));
        assert!(prompt
            .user
            .contains("retrievals: UsersKB -> prefers concise answers"));
        // The supplied timestamp is injected for the model to reuse.
        assert!(prompt.user.contains("2026-06-16T12:00:00Z"));
    }

    #[test]
    fn call2_empty_context_becomes_a_placeholder() {
        let prompt = Call2Prompt::build("threadkeeper", "hi", "   ", "2026-06-16T12:00:00Z");
        assert!(prompt
            .user
            .contains("(no research results were assembled this turn)"));
    }

    #[test]
    fn call2_system_lists_both_documents_and_the_output_rule() {
        let prompt = Call2Prompt::build("threadkeeper", "hi", "", "2026-06-16T12:00:00Z");

        for document in ["response.json", "actions.json"] {
            assert!(prompt.system.contains(document), "missing {document}");
        }
        // The single-array output rule must be present and unambiguous.
        assert!(prompt.system.contains("single JSON array"));
    }

    #[test]
    fn call2_system_lists_every_side_effecting_action() {
        let prompt = Call2Prompt::build("threadkeeper", "hi", "", "2026-06-16T12:00:00Z");

        for effect in [
            "upsert",
            "send",
            "actuate",
            "connection",
            "WhatsApp",
            "Push",
            "Actuator",
            "GaiaConnections",
        ] {
            assert!(prompt.system.contains(effect), "missing {effect}");
        }
    }
}
