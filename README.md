# Gaia Robot

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fthreadkeeper%2Fgaia-robot%2Fmain%2Finfra%2Fazuredeploy.json)

> The **Deploy to Azure** button provisions the Cosmos DB for NoSQL account with
> vector-search and full-text-search capabilities and the `gaia` database, an
> Azure AI Foundry account/project with the `model-router` model, and a
> Container App (see [infra/azuredeploy.json](infra/azuredeploy.json)). The
> default **Free / Lite** tier has **no fixed idle cost** — see the full
> [cost / pricing breakdown](infra/README.md#cost--pricing-breakdown). After it
> completes, copy the `cosmosEndpoint` output into `infra/.env` and run
> [infra/cosmos_create.py](infra/cosmos_create.py) to create the six containers.

A small Rust console application, structured for **simplicity, clarity, and
strong supply-chain hygiene**. See [.github/copilot-instructions.md](.github/copilot-instructions.md)
for the full coding standards that govern this repository.

## Repository Layout

| Folder       | Purpose                                                              |
|--------------|---------------------------------------------------------------------|
| `rust/`      | All Rust source code (the Cargo project lives here).                 |
| `infra/`     | Infrastructure, deployment, and environment configuration.          |
| `tests/`     | Top-level integration and end-to-end tests for the whole program.   |
| `app/`       | Application-level assets, configuration, and runtime resources.     |
| `research/`  | Experiments, prototypes, notes, and exploratory work.               |

## Program Architecture

The program is modeled like a C# console app:

- **`rust/src/main.rs`** is a single, well-commented orchestrator. Read it
  top-to-bottom to follow the entire program flow.
- Each "class" lives in its own file (one type per module), named after the
  type. `main.rs` wires these pieces together; it holds no detailed logic.

Current modules:

- [`rust/src/command.rs`](rust/src/command.rs) — the `Command` type (parses user input).
- [`rust/src/robot.rs`](rust/src/robot.rs) — the `Robot` type (decides responses).

## Prerequisites

Install Rust via [rustup](https://rustup.rs/). The toolchain is pinned in
[`rust/rust-toolchain.toml`](rust/rust-toolchain.toml), so rustup will select
the correct compiler automatically.

## Building & Running

All Cargo commands run from the `rust/` folder:

```sh
cd rust
cargo run            # build and start the console app
cargo build --release
```

Type `hello`, `name`, `echo <text>`, or `quit` at the prompt.

## Quality Gate

These checks are enforced in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))
on every push and pull request. Run them locally from `rust/` before pushing:

```sh
cargo fmt --all -- --check                          # formatting
cargo clippy --all-targets --all-features -- -D warnings   # lint (warnings = errors)
cargo test --all-features                           # unit + integration tests
cargo llvm-cov --all-features --fail-under-lines 80 # coverage gate
cargo audit                                         # known vulnerabilities (RustSec)
cargo deny check                                    # advisories, licenses, bans, sources
```

`cargo-llvm-cov`, `cargo-audit`, and `cargo-deny` are extra tools; install them
once with:

```sh
cargo install cargo-llvm-cov cargo-audit cargo-deny
```

### Fast local feedback

Enable the pre-commit hook (formatting + lint + tests) once per clone:

```sh
git config core.hooksPath .githooks
```

## Testing

- Unit tests live beside the type they cover, in a `#[cfg(test)] mod tests`
  block within each module file.
- Integration tests that drive the built binary live in
  [`rust/tests/`](rust/tests/) (see `cli.rs`).
- Every piece of program logic must be covered by at least one test.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
