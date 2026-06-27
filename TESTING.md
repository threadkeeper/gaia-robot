# Testing Gaia

Gaia is tested at several levels, all enforced automatically by CI
([.github/workflows/ci.yml](.github/workflows/ci.yml)). This document explains
every layer, what it covers, and how to run it yourself.

All Cargo commands run from the **`rust/`** folder, where the Cargo project
lives:

```sh
cd rust
```

---

## 1. Unit tests (326)

Every type lives in its own module with a `#[cfg(test)] mod tests` block beside
it, so the logic and its tests sit together. These are fast, hermetic, and need
no cloud credentials — they cover parsing, routing, the two-pass thought
sequence, the data controllers, static-file serving, and more.

```sh
cargo test --all-features
```

The same command also runs the integration tests below.

---

## 2. Integration tests (3)

[`rust/tests/cli.rs`](rust/tests/cli.rs) launches the **compiled binary**, feeds
it scripted input on stdin, and asserts on its output. It proves `main` walks all
eleven program-flow blocks top-to-bottom and honours the Esc / end-of-input quit
signals:

- `walks_all_eleven_blocks_in_one_pass`
- `quits_immediately_when_esc_pressed_at_the_user_block`
- `quits_cleanly_on_end_of_input`

Cargo discovers these automatically from `rust/tests/`, so `cargo test` runs them
alongside the unit tests. The top-level [`tests/`](tests/) folder holds the
fixtures the live self-tests read and write (see §5) plus a placeholder for
future cross-service end-to-end tests.

---

## 3. Coverage gate (≥ 80 %)

Line coverage is measured with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)
and the build fails if it drops below 80 %:

```sh
cargo llvm-cov --all-features --fail-under-lines 80
```

In CI the coverage job runs on every push and uploads an `lcov.info` artifact. It
is a tracked quality signal but is intentionally **not** allowed to block a
deploy.

---

## 4. Supply-chain audit

Every third-party crate is treated as a potential attack vector, so two checks
run on each push:

```sh
cargo audit       # known vulnerabilities (RustSec advisory DB)
cargo deny check  # advisories, license, ban, and source policy (rust/deny.toml)
```

Install all the extra tooling once with
`cargo install cargo-llvm-cov cargo-audit cargo-deny`.

---

## 5. Live self-tests (deploy gates)

Three opt-in subcommands exercise the **real** Foundry model and **real** Cosmos
account end-to-end. They exit non-zero on any failure, so each doubles as an
on-demand check and a hard CI deploy gate. All require `GAIA_MODE=dev` (the LLM
and Cosmos clients are disabled otherwise) plus the live `FOUNDRY_*` / `COSMOS_*`
configuration.

| Subcommand | What it proves | Fixtures |
|------------|----------------|----------|
| `gaia-robot test-data-retrieval` | The **pull pass** (LLM Call 1): asks the model five questions, parses each `actions.json`, and runs the resulting Cosmos queries + Brave web search. | writes [`tests/LLM1`](tests/LLM1) |
| `gaia-robot test-data-execution` | The **push pass** (LLM Call 2): replays the five contexts captured under `tests/LLM1` and audits that every required side-effect record (reply, WhatsApp, writes…) was emitted. | reads `tests/LLM1`, writes [`tests/LLM2`](tests/LLM2) |
| `gaia-robot test-data-persistence` | The **write pass**: appends two chunks per container through the shared `WriteDataController`, reads each back, and verifies the append, the refreshed embedding, and the synced `DataLakeIndex` vector. | live Cosmos round-trip |

Run one locally (after filling in `infra/.env` or exporting the variables):

```sh
cd rust
GAIA_MODE=dev cargo run --quiet -- test-data-retrieval
```

### As a CI deploy gate

The `data-retrieval-smoke` job runs `test-data-retrieval` on pushes to `main`,
authenticating to Azure via OIDC and minting a Cosmos data-plane AAD token. The
`deploy` job `needs` it, so a red self-test **blocks** the rollout to the
Container App. See the job comments in
[.github/workflows/ci.yml](.github/workflows/ci.yml) for the exact secrets and
variables it expects.

---

## Run everything CI runs

```sh
cd rust
cargo fmt --all -- --check                                 # formatting
cargo clippy --all-targets --all-features -- -D warnings   # lint (warnings = errors)
cargo test --all-features                                  # unit + integration
cargo llvm-cov --all-features --fail-under-lines 80         # coverage gate
cargo audit                                                # known vulnerabilities
cargo deny check                                           # licenses, bans, sources
```

For fast local feedback, enable the pre-commit hook once per clone with
`git config core.hooksPath .githooks`.
