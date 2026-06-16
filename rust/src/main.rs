//! Gaia robot console application.
//!
//! This is the single orchestrating entry point for the program. Read it
//! top-to-bottom to follow the entire flow, much like a C# console app's
//! `Main` method. The detailed data of each "class" lives in its own module
//! (one type per file); `main` only wires those pieces together.
//!
//! The program walks the eleven blocks of the architecture diagram
//! (`Gaia Physical Architecture.drawio.png`) as a loop:
//!
//! 1. The **User** block prompts for input.
//! 2. Each remaining block logs its exact description and waits for Enter.
//! 3. After the last block the loop restarts at the **User** block.
//!
//! The loop runs until the user presses Esc (or input reaches end-of-file).

// Each module holds exactly one primary type plus its tests.
mod flow;

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

/// The Esc key. When this control character appears in a line of input we treat
/// it as the signal to stop the program.
///
/// Note: a portable, std-only console reads input a line at a time, so the user
/// presses Esc and then Enter. Detecting a bare single Esc keystroke would
/// require putting the terminal into raw mode (e.g. via the `crossterm` crate),
/// which this skeleton intentionally avoids to stay dependency-free.
const ESC: char = '\u{1b}';

/// Program entry point.
///
/// Returns [`ExitCode::SUCCESS`] on a clean exit and [`ExitCode::FAILURE`] if
/// the program cannot read from or write to its standard streams. We return an
/// `ExitCode` rather than panicking so failures surface as a normal, testable
/// exit status.
fn main() -> ExitCode {
    // --- 1. Set up the flow steps and the I/O streams ----------------------
    // The flow module owns the eleven blocks; `main` just drives them in order.
    let steps = flow::steps();

    // Lock stdin/stdout once up front for efficient line-by-line interaction.
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    // --- 2. Print the banner -----------------------------------------------
    // `writeln!` can fail if stdout is closed; treat that as a fatal error.
    if writeln!(
        output,
        "Gaia program flow. Press Esc then Enter at any prompt to quit.",
    )
    .is_err()
    {
        return ExitCode::FAILURE;
    }

    // --- 3. Run the program-flow loop --------------------------------------
    // Each outer iteration walks all eleven blocks once. We reuse a single
    // input buffer across reads to avoid needless allocations.
    let mut line = String::new();
    'flow: loop {
        for (index, step) in steps.iter().enumerate() {
            // Log the block's exact description, mirroring the diagram.
            if writeln!(output, "\n=== {} ===", step.title()).is_err()
                || writeln!(output, "{}", step.description()).is_err()
            {
                return ExitCode::FAILURE;
            }

            // The first block (User) prompts for input; every other block just
            // waits for Enter before moving to the next part of the flow.
            let prompt = if index == 0 {
                "Your input> "
            } else {
                "Press Enter to continue (Esc to quit)> "
            };
            if write!(output, "{prompt}").is_err() || output.flush().is_err() {
                return ExitCode::FAILURE;
            }

            // Read one line of input to advance the flow.
            line.clear();
            match input.read_line(&mut line) {
                // End of input (e.g. piped file or Ctrl-Z/Ctrl-D): exit cleanly.
                Ok(0) => break 'flow,
                Ok(_) => {}
                // Could not read input at all: nothing sensible to do but fail.
                Err(_) => return ExitCode::FAILURE,
            }

            // Esc anywhere in the line ends the whole loop, not just this block.
            if line.contains(ESC) {
                break 'flow;
            }
        }
    }

    // --- 4. Exit -----------------------------------------------------------
    if writeln!(output, "\nGoodbye!").is_err() {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
