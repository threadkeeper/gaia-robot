//! End-to-end integration test for the Gaia robot CLI.
//!
//! Unlike the unit tests inside each module, this test launches the actual
//! compiled binary, feeds it a script of input lines on standard input, and
//! checks the combined output. It verifies that `main` walks the eleven
//! program-flow blocks and honors the Esc / end-of-input quit signals.

use std::io::Write;
use std::process::{Command, Stdio};

/// The Esc control character, used to tell the program to quit.
const ESC: &str = "\u{1b}";

/// Run the compiled `gaia-robot` binary, piping `input` to its stdin, and
/// return everything it wrote to stdout as a single `String`.
fn run_with_input(input: &str) -> String {
    // Cargo sets CARGO_BIN_EXE_<name> to the path of the built binary for
    // integration tests, so we don't have to guess where `target/` is.
    let exe = env!("CARGO_BIN_EXE_gaia-robot");

    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to start gaia-robot");

    // Write the scripted input, then drop stdin so the child sees EOF.
    child
        .stdin
        .take()
        .expect("child stdin was not captured")
        .write_all(input.as_bytes())
        .expect("failed to write to child stdin");

    let output = child
        .wait_with_output()
        .expect("failed to wait for gaia-robot");

    assert!(output.status.success(), "process exited with failure");
    String::from_utf8(output.stdout).expect("stdout was not valid UTF-8")
}

#[test]
fn walks_all_eleven_blocks_in_one_pass() {
    // One line of user input for the User block, then ten Enters (empty lines)
    // for the remaining blocks, then Esc to stop before a second pass.
    let mut script = String::from("what is the weather\n");
    script.push_str(&"\n".repeat(10));
    script.push_str(&format!("{ESC}\n"));

    let output = run_with_input(&script);

    // Startup banner.
    assert!(output.contains("Gaia program flow"));
    // A sampling of block descriptions, proving each part of the flow is logged.
    assert!(output.contains("Isaac Asimov"));
    assert!(output.contains("LLM Call 1"));
    assert!(output.contains("send WhatsApp / value"));
    assert!(output.contains("analysis.json"));
    assert!(output.contains("Response Data Context"));
    assert!(output.contains("LLM Call 2"));
    assert!(output.contains("response.json"));
    // Clean farewell after quitting.
    assert!(output.contains("Goodbye!"));
}

#[test]
fn quits_immediately_when_esc_pressed_at_the_user_block() {
    // Esc at the very first prompt should stop before reaching LLM Call 1.
    let output = run_with_input(&format!("{ESC}\n"));

    assert!(output.contains("Gaia program flow"));
    assert!(output.contains("Goodbye!"));
    assert!(!output.contains("LLM Call 1"));
}

#[test]
fn quits_cleanly_on_end_of_input() {
    // No input at all: the loop should end when stdin closes (EOF).
    let output = run_with_input("");
    assert!(output.contains("Goodbye!"));
}
