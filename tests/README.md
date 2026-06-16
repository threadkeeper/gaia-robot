# tests/

Top-level integration and end-to-end tests that exercise the program as a
whole (e.g. cross-language or cross-service scenarios, scripted CLI runs
against a built artifact).

Note: Cargo also runs Rust integration tests from `rust/tests/`, since that is
where Cargo discovers them automatically (see `rust/tests/cli.rs`, which drives
the compiled binary end-to-end). Use this top-level folder for higher-level or
non-Rust end-to-end tests.

This folder is intentionally a placeholder for now.
