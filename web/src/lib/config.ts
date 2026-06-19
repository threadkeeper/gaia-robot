/**
 * Runtime configuration resolved from Vite env vars (see .env.example).
 * Only VITE_-prefixed vars are exposed to the browser bundle.
 */

const env = import.meta.env;

/** Base URL of the Gaia API. Empty string => same-origin (prod) / dev proxy. */
export const API_BASE: string = (env.VITE_API_BASE ?? '').replace(/\/$/, '');

/** Preferred reply transport: 'ws' streams over a WebSocket, 'post' is a single non-streaming POST. */
export const STREAM_TRANSPORT: 'ws' | 'post' =
  env.VITE_STREAM_TRANSPORT === 'post' ? 'post' : 'ws';

/** Direct Google sign-in client id (Google Identity Services). */
export const GOOGLE_CLIENT_ID: string = (env.VITE_GOOGLE_CLIENT_ID ?? '') as string;

/** Whether Google sign-in is configured in this build. */
export const GOOGLE_CONFIGURED = !!GOOGLE_CLIENT_ID;

/** True when no Google client id is configured: run in local dev-auth mode. */
export const DEV_AUTH = !GOOGLE_CONFIGURED;

/** Build an absolute API URL from a path. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

/** Build an absolute WebSocket URL from a path. */
export function wsUrl(path: string): string {
  if (API_BASE) {
    return `${API_BASE.replace(/^http/, 'ws')}${path.startsWith('/') ? path : `/${path}`}`;
  }
  // Same-origin: derive ws(s) from the current page.
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}${path.startsWith('/') ? path : `/${path}`}`;
}
