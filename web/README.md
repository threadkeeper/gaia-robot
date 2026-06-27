# Gaia PWA

The installable **Progressive Web App** front end for Gaia — a SvelteKit + TypeScript
client for the conversation gateway. It renders a streaming chat UI,
signs the user into their private wing, and can be installed to the home screen.

The backend it talks to is the Rust app in [`../rust`](../rust); in production the
**same** Container App image serves both this PWA and the API from one origin.

## Stack

- **SvelteKit** (Svelte 5 runes) + **TypeScript**
- **adapter-static** — the shell is prerendered as an SPA so Azure Front Door can cache it
- **@vite-pwa/sveltekit** (Workbox) — service worker + web app manifest (installable, offline shell)

## Quick start

```bash
cd web
npm install
cp .env.example .env      # optional; defaults work against a local backend
npm run dev               # http://localhost:5173
```

By default the dev server proxies `/v1`, `/healthz`, `/readyz` (and the WebSocket)
to a Gaia backend on `http://localhost:80`. Start it from the repo root:

```bash
# from ../rust — run the Rust backend in HTTP server mode
cd ../rust
GAIA_MODE=dev GAIA_HTTP_PORT=80 cargo run
```

Override the proxy target with `VITE_API_PROXY`, or point straight at a deployed
backend with `VITE_API_BASE=https://<app>.azurecontainerapps.io`.

## Authentication

- **Local (default):** no Google client id configured → "dev auth" mode. You pick a display name and
  the app sends `Authorization: Bearer dev:<name>`; the Rust backend accepts a bearer subject
  for local use (`rust/src/auth.rs`).
- **Production:** set `VITE_GOOGLE_CLIENT_ID` for direct Google sign-in via Google Identity
  Services (`src/lib/auth/google.ts`). The browser receives a Google ID token credential,
  exchanges it at `POST /v1/auth/google`, and stores the returned Gaia session JWT.

  All conversation API calls use the Gaia session JWT as the Bearer credential. The backend
  validates it locally and maps `sub` → a private wing.

## API contract consumed

| Transport | Endpoint | Notes |
|---|---|---|
| WS   | `/v1/ws/{id}` | default; hello `{token}` then `{text}`, streams `{type:'token'}`/`{type:'done'}` |
| POST | `/v1/conversations/{id}/messages` | non-streaming; set `VITE_STREAM_TRANSPORT=post` |
| GET  | `/healthz` | liveness (cheap) |
| GET  | `/readyz` | readiness (deep dependency + RBAC probe) |

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | dev server with API proxy |
| `npm run build` | production static build → `build/` |
| `npm run preview` | preview the production build |
| `npm run check` | `svelte-check` type checking |
| `npm run lint` / `npm run format` | ESLint + Prettier |

## Build output & deployment

`npm run build` emits a static site in `build/` (plus the generated service worker and
manifest). In CI/CD the image is built from the **repository root** with the multi-stage
[`Dockerfile`](../Dockerfile): a Node stage compiles this app, and the Rust backend
serves the resulting `build/` as static files (with SPA fallback) from `GAIA_WEB_DIR`.
So a single container image hosts both the API and the installable PWA; `web/**` changes
trigger the CD workflow, which re-bakes the `VITE_*` values into the bundle on each build.

## Icons

App icons are PNGs generated from `../gaia.png` into [`static/icons/`](static/icons)
(192/512 `any` + maskable, a 180px apple-touch icon, and `favicon.ico`). Regenerate them
with Pillow if the source art changes.
