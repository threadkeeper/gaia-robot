//! The [`FlowStep`] type: one block of Gaia's program-flow diagram.
//!
//! The architecture PNG (`Gaia Physical Architecture.drawio.png`) describes the
//! program as a loop of eleven blocks, starting at the **User** block and
//! looping back through **response.json**. Each [`FlowStep`] captures one of
//! those blocks: a short `title` (the block's name) and the exact `description`
//! text shown inside the block. `main` walks these steps in order, logging each
//! description, to give the running program the same shape as the diagram.

/// A single block in Gaia's program-flow diagram.
///
/// A step is just data: a human-readable `title` and the verbatim block
/// `description`. The first step in [`steps`] is the **User** input block; the
/// program prompts for input there. Every other step is logged and then waits
/// for the user to press Enter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowStep {
    /// The block's short name, e.g. `"User"` or `"LLM Call 1"`.
    title: &'static str,
    /// The exact text shown inside the block on the architecture diagram.
    description: &'static str,
}

impl FlowStep {
    /// The block's short name (e.g. `"User"`).
    pub fn title(&self) -> &str {
        self.title
    }

    /// The verbatim text shown inside the block on the diagram.
    pub fn description(&self) -> &str {
        self.description
    }
}

/// The eleven program-flow blocks, in execution order.
///
/// The order mirrors the arrows in the architecture diagram: the loop starts at
/// the **User** block (index 0, where input is collected), flows through the two
/// LLM calls and their JSON outputs, and ends at **response.json**, which feeds
/// back to the user to begin the next iteration.
///
/// # Examples
///
/// ```ignore
/// // (internal module; shown for illustration)
/// let steps = flow::steps();
/// assert_eq!(steps.len(), 11);
/// assert_eq!(steps[0].title(), "User");
/// ```
pub fn steps() -> Vec<FlowStep> {
    vec![
        // 1. User — the only block that collects input rather than waiting for Enter.
        FlowStep {
            title: "User",
            description: "Prompt the user for their question / input.",
        },
        // 2. Gaia context note feeding LLM Call 1.
        FlowStep {
            title: "Gaia Context",
            description: "Gaia, robot from the future featured in Isaac Asimov's \
world, save humanity intelligence and people.\n\nContext from Cosmos max 100k",
        },
        // 3. First LLM call: analyze the question and emit the analysis artifacts.
        FlowStep {
            title: "LLM Call 1",
            description: "LLM Call 1\nAnalyze the question and output\n\
actions.json {}\nanalysis.json {}\nfacts.json {}\nnewContext.json {}",
        },
        // 4. actions.json produced by LLM Call 1 — the read-only GET actions.
        FlowStep {
            title: "actions.json",
            description: "actions.json\nGET\nWeb / value\n\
users dl semantic Index name / value\nusers kb semantic Index name / value\n\
gaia kb semantic Index name / value\ngaia lh logical Index name / value\n\
gaia cosmos index named / value",
        },
        // 5. analysis.json produced by LLM Call 1.
        FlowStep {
            title: "analysis.json",
            description: "analysis.json\nemotion / value\ntruthfulness / value\n\
intention / value",
        },
        // 6. facts.json produced by LLM Call 1.
        FlowStep {
            title: "facts.json",
            description: "facts.json\n[{fact / value}]",
        },
        // 7. newContext.json produced by LLM Call 1.
        FlowStep {
            title: "newContext.json",
            description: "newContext.json\nOld Context * 0.61",
        },
        // 8. The combined response data + context (the wavy banner in the diagram).
        FlowStep {
            title: "Response Data Context",
            description: "Response Data Context",
        },
        // 9. Second LLM call: turn the response data into final actions + response.
        FlowStep {
            title: "LLM Call 2",
            description: "LLM Call 2\nAnalyze the question and output\n\
actions.json\nresponse.json",
        },
        // 10. actions.json produced by LLM Call 2 — the side-effecting POST actions.
        FlowStep {
            title: "actions.json",
            description: "actions.json\nPOST\nSearch Web / value\n\
users dl semantic Index name / value\nusers kb semantic Index name / value\n\
gaia kb semantic Index name / value\ngaia lh logical Index name / value\n\
gaia cosmos index named / value\nsend WhatsApp / value\nsend Push / value\n\
actuate / value\nconnection / value",
        },
        // 11. response.json delivered back to the user, closing the loop.
        FlowStep {
            title: "response.json",
            description: "response.json\nData",
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_exactly_eleven_blocks() {
        assert_eq!(steps().len(), 11);
    }

    #[test]
    fn loop_starts_at_user_and_ends_at_response_json() {
        let steps = steps();
        assert_eq!(steps.first().unwrap().title(), "User");
        assert_eq!(steps.last().unwrap().title(), "response.json");
    }

    #[test]
    fn block_titles_follow_the_diagram_order() {
        // Bind `steps()` first so the Vec outlives the borrowed `&str` titles.
        let steps = steps();
        let titles: Vec<&str> = steps.iter().map(FlowStep::title).collect();
        assert_eq!(
            titles,
            vec![
                "User",
                "Gaia Context",
                "LLM Call 1",
                "actions.json",
                "analysis.json",
                "facts.json",
                "newContext.json",
                "Response Data Context",
                "LLM Call 2",
                "actions.json",
                "response.json",
            ]
        );
    }

    #[test]
    fn llm_call_one_lists_its_four_json_outputs() {
        let llm_call_1 = &steps()[2];
        let description = llm_call_1.description();
        assert!(description.contains("actions.json"));
        assert!(description.contains("analysis.json"));
        assert!(description.contains("facts.json"));
        assert!(description.contains("newContext.json"));
        // connection moved to the LLM Call 2 POST actions, so it is not here.
        assert!(!description.contains("connection"));
    }

    #[test]
    fn call_one_actions_are_read_only_gets() {
        // Block 3 is the GET (pull) action list emitted by LLM Call 1.
        let get_actions = &steps()[3];
        let description = get_actions.description();
        assert!(description.contains("GET"));
        assert!(description.contains("gaia cosmos index named / value"));
        // Side-effecting actions belong to the Call 2 POST block, not here.
        assert!(!description.contains("send WhatsApp"));
        assert!(!description.contains("actuate"));
        assert!(!description.contains("connection"));
    }

    #[test]
    fn call_two_actions_post_side_effects_including_connection() {
        // Block 9 is the POST (push) action list emitted by LLM Call 2.
        let post_actions = &steps()[9];
        let description = post_actions.description();
        assert!(description.contains("POST"));
        assert!(description.contains("send WhatsApp / value"));
        assert!(description.contains("send Push / value"));
        assert!(description.contains("actuate / value"));
        assert!(description.contains("connection / value"));
    }
}
