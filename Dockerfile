# syntax=docker/dockerfile:1

# Multi-stage build for the Gaia backend (the Rust HTTP/WebSocket server).
#
# Stage 1 compiles an optimized, statically-linked-enough release binary; stage
# 2 copies just that binary into a slim Debian runtime so the shipped image stays
# small and has no build tooling or source in it.

# ---- Build stage: compile the release binary ----
FROM rust:1-bookworm AS builder
WORKDIR /build

# The Cargo project lives in rust/. Copy it in and build with the committed
# Cargo.lock (`--locked`) so the image matches CI exactly. ring (pulled in via
# ureq's rustls TLS) needs a C toolchain, which the full rust:bookworm image
# already provides.
COPY rust/ ./
RUN cargo build --release --locked

# ---- Runtime stage: minimal image with just the binary ----
FROM debian:bookworm-slim AS runtime

# Run as a dedicated non-root user.
RUN useradd --create-home --uid 10001 gaia
USER gaia

# Copy only the compiled binary from the builder.
COPY --from=builder /build/target/release/gaia-robot /usr/local/bin/gaia-robot

# The Container App ingress targets port 80; bind the HTTP server there. The
# server only starts when GAIA_HTTP_ADDR/GAIA_HTTP_PORT is set, so this env var
# is what flips the binary from the interactive console into server mode.
ENV GAIA_HTTP_PORT=80
EXPOSE 80

ENTRYPOINT ["/usr/local/bin/gaia-robot"]
