# Project Coding Standards

This is a Rust project. Always write idiomatic, safe Rust and favor **simplicity and clarity** over cleverness.

## Core Principles

- **Favor simplicity.** Prefer the most straightforward solution that works. Avoid premature abstraction, over-engineering, and unnecessary generics or trait gymnastics.
- **Comment generously.** Explain the *why* behind non-obvious code, not just the *what*. Every public item (`pub fn`, `pub struct`, `pub enum`, `pub trait`) must have a `///` doc comment. Add inline `//` comments to clarify tricky logic, invariants, and assumptions.
- **Readable over terse.** Use clear, descriptive names. A few extra lines that read well beat a dense one-liner.

## Project Structure

Organize the repository into these top-level folders:

- `rust/` — all Rust source code (the Cargo project lives here).
- `infra/` — infrastructure, deployment, and environment configuration.
- `tests/` — integration and end-to-end tests that exercise the program as a whole.
- `app/` — application-level assets, configuration, and runtime resources.
- `research/` — experiments, prototypes, notes, and exploratory work kept separate from production code.

## Program Architecture

- **Single orchestrating `main`.** Model the program like a C# console app: one long, well-commented `main.rs` that reads top-to-bottom and drives the entire program flow. A reader should be able to follow the whole program by reading `main.rs`.
- **One type per file.** Put each "class" (struct/enum plus its `impl` blocks) in its own separate file/module, named after the type. `main.rs` wires these pieces together; it does not contain the detailed logic of each type.
- **Keep `main` readable.** `main.rs` may be long, but it should stay a clear sequence of high-level steps that delegate to the type modules. The "split into small helpers" guidance applies *inside* each type's module, not to `main`'s overall flow.

## Rust Conventions

- Target stable Rust and follow the official [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Format all code with `rustfmt` (default settings).
- Code must pass `cargo clippy` with no warnings. Fix lints rather than suppressing them; only `#[allow(...)]` with a comment explaining why.
- Prefer `Result<T, E>` and the `?` operator for error handling. Avoid `unwrap()` and `expect()` outside of tests, examples, or cases where failure is truly impossible (and document why).
- Avoid `unsafe`. If unavoidable, isolate it, keep it minimal, and add a `// SAFETY:` comment justifying each use.
- Use `&str`/`&[T]` parameters over owned `String`/`Vec<T>` when ownership isn't needed.
- Prefer iterators and standard combinators over manual index loops when it improves clarity.
- Derive common traits (`Debug`, `Clone`, `PartialEq`, etc.) where reasonable.

## Dependencies & Supply-Chain Security

Treat every third-party crate as a potential supply-chain attack vector. Be conservative and deliberate about dependencies.

- **Prefer the standard library.** Do not add a dependency for something `std` already does well.
- **Justify new crates.** Before adding a dependency, confirm it is widely used, actively maintained, and from a reputable source. Note the reason in the PR/commit.
- **No stale dependencies.** Do not use crates that are unmaintained, yanked, or significantly behind their latest stable release. Keep `Cargo.toml` versions current.
- **Pin and lock.** Commit `Cargo.lock`. Use explicit, sensible version requirements rather than wildcards.
- **Audit regularly.** Code should pass `cargo audit` (no known vulnerabilities) and ideally `cargo deny` checks. Run these before adding or updating dependencies.
- **Minimize the tree.** Prefer crates with few transitive dependencies. Avoid pulling in large dependency trees for small features.

## Comments & Documentation

- Doc comments (`///`) on all public APIs, including a short summary and, where helpful, an `# Examples` section.
- Module-level `//!` comments describing the purpose of each module.
- Use inline comments to explain intent, edge cases, and any non-obvious decisions.
- Keep comments accurate—update them when the code changes.

## Testing

- **Full coverage of logic.** Every piece of program logic must be covered by at least one test. No logic ships without a corresponding test.
- Add unit tests in a `#[cfg(test)] mod tests` block alongside the type they cover (in that type's module file).
- Put integration and end-to-end tests in the top-level `tests/` folder.
- Write small, focused tests with clear names that describe the scenario.
- When adding or changing logic, add or update its tests in the same change.

## What to Avoid

- Deep nesting—prefer early returns and `?` to flatten control flow.
- Clever or obscure constructs when a simple alternative exists.
- Large functions—split them into small, well-named helpers.
- Adding dependencies for trivial functionality that the standard library already covers.

## Enforcement & Tooling

These standards are enforced automatically. Run them locally before pushing (all from the `rust/` folder):

- `cargo fmt --all -- --check` — formatting.
- `cargo clippy --all-targets --all-features -- -D warnings` — lint, warnings are errors.
- `cargo test --all-features` — tests.
- `cargo llvm-cov --all-features --fail-under-lines 80` — coverage gate.
- `cargo audit` — known vulnerabilities (RustSec advisory DB).
- `cargo deny check` — advisories, license, ban, and source policy (see `rust/deny.toml`).

CI runs all of the above on every push and pull request (`.github/workflows/ci.yml`). For fast local feedback, enable the pre-commit hook once per clone with `git config core.hooksPath .githooks`.
