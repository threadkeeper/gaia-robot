# syntax=docker/dockerfile:1

# Multi-stage build for Gaia: one image serves BOTH the JSON/WebSocket API and
# the installable PWA front end, so a single Container App hosts everything on one
# origin (no CORS, no separate Static Web App).
#
# Stage 1 builds the SvelteKit PWA to static files. The public sign-in client ids
# are inlined into the bundle at THIS step via VITE_* build args, which is how the
# CI/CD pipeline injects per-environment values on each build.
# Stage 2 compiles the Rust backend to an optimized release binary.
# Stage 3 copies just the binary and the built web assets into a slim runtime.

# ---- Web stage: build the PWA static bundle ----
FROM node:20-bookworm-slim AS web
WORKDIR /web

# VITE_* are inlined at build time (see web/src/lib/config.ts). They are public
# identifiers (client ids / transport), safe to bake into the browser bundle.
# VITE_API_BASE is intentionally left empty: the API is served from the same
# origin as the PWA, so the front end uses relative paths.
ARG VITE_GITHUB_CLIENT_ID=""
ARG VITE_GOOGLE_CLIENT_ID=""
ARG VITE_STREAM_TRANSPORT=""
ENV VITE_GITHUB_CLIENT_ID=$VITE_GITHUB_CLIENT_ID \
    VITE_GOOGLE_CLIENT_ID=$VITE_GOOGLE_CLIENT_ID \
    VITE_STREAM_TRANSPORT=$VITE_STREAM_TRANSPORT

# Install dependencies against the committed lockfile first (better layer cache),
# then build the static site into /web/build.
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Build stage: compile the release binary ----
FROM rust:1-bookworm AS builder
WORKDIR /build

# The Cargo project lives in rust/. Copy it in and build with the committed
# Cargo.lock (`--locked`) so the image matches CI exactly. ring (pulled in via
# ureq's rustls TLS) needs a C toolchain, which the full rust:bookworm image
# already provides.
COPY rust/ ./
RUN cargo build --release --locked

# ---- Runtime stage: minimal image with the binary + web assets ----
FROM debian:bookworm-slim AS runtime

# Run as a dedicated non-root user.
RUN useradd --create-home --uid 10001 gaia
USER gaia

# Copy the compiled binary and the built PWA.
COPY --from=builder /build/target/release/gaia-robot /usr/local/bin/gaia-robot
COPY --from=web /web/build /usr/local/share/gaia-web

# The Container App ingress targets port 80; bind the HTTP server there. The
# server only starts when GAIA_HTTP_ADDR/GAIA_HTTP_PORT is set, so this env var
# is what flips the binary from the interactive console into server mode.
# GAIA_WEB_DIR points the server at the bundled PWA so it serves the web app too.
ENV GAIA_HTTP_PORT=80 \
    GAIA_WEB_DIR=/usr/local/share/gaia-web
EXPOSE 80

ENTRYPOINT ["/usr/local/bin/gaia-robot"]
