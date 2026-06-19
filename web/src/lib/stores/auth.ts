/**
 * Authentication store.
 *
 * In production the frontend collects a Google ID token (GIS), exchanges it at
 * POST /v1/auth/google, and stores the returned Gaia session JWT. For local
 * development — when no Google client id is configured — it runs a "dev auth"
 * mode where the user picks a subject and sends it as the bearer token.
 */
import { browser } from '$app/environment';
import { exchangeGoogleToken, refreshSession } from '$lib/api/client';
import { GOOGLE_CONFIGURED } from '$lib/config';
import type { AuthUser } from '$lib/types';
import { writable } from 'svelte/store';

const STORAGE_KEY = 'gaia.auth.v1';
const MODE_KEY = 'gaia.auth.mode.v1';

/** Refresh the access token this many seconds before it actually expires. */
const REFRESH_SKEW_SECONDS = 60;

export type AuthMode = 'google' | 'dev';

function defaultMode(): AuthMode {
  return GOOGLE_CONFIGURED ? 'google' : 'dev';
}

function load(): AuthUser | null {
  if (!browser) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

function persist(user: AuthUser | null): void {
  if (!browser) return;
  try {
    if (user) localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore quota / private mode */
  }
}

function loadMode(): AuthMode {
  if (!browser) return defaultMode();
  if (!GOOGLE_CONFIGURED) return 'dev';
  try {
    const raw = localStorage.getItem(MODE_KEY);
    return raw === 'dev' || raw === 'google' ? raw : 'google';
  } catch {
    return 'google';
  }
}

function persistMode(mode: AuthMode): void {
  if (!browser) return;
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    /* ignore quota / private mode */
  }
}

const _authMode = writable<AuthMode>(loadMode());
let _mode: AuthMode = loadMode();
_authMode.subscribe((m) => {
  _mode = !GOOGLE_CONFIGURED ? 'dev' : m;
  persistMode(_mode);
});

export const authMode = {
  subscribe: _authMode.subscribe
};

function createAuth() {
  const { subscribe, set } = writable<AuthUser | null>(load());

  let current: AuthUser | null = load();
  subscribe((u) => {
    current = u;
  });

  /** True when a Google-mode access token is missing or within the skew window. */
  function accessTokenStale(user: AuthUser): boolean {
    if (user.dev) return false;
    if (!user.expiresAt) return true;
    return user.expiresAt - REFRESH_SKEW_SECONDS <= Math.floor(Date.now() / 1000);
  }

  /** Attempt a silent refresh using the stored refresh token. */
  let refreshInFlight: Promise<string | null> | null = null;
  async function doRefresh(user: AuthUser): Promise<string | null> {
    if (!user.refreshToken) return null;
    if (refreshInFlight) return refreshInFlight;
    refreshInFlight = (async () => {
      try {
        const res = await refreshSession(user.refreshToken as string);
        const next: AuthUser = {
          ...user,
          sub: res.user.sub || user.sub,
          token: res.token,
          expiresAt: res.expiresAt,
          refreshToken: res.refreshToken,
          dev: false
        };
        persist(next);
        set(next);
        return next.token;
      } catch {
        // Refresh token expired or invalid — force a clean re-sign-in.
        persist(null);
        set(null);
        return null;
      } finally {
        refreshInFlight = null;
      }
    })();
    return refreshInFlight;
  }

  return {
    subscribe,

    /** Capability flags used by the sign-in UI. */
    canUseGoogle: GOOGLE_CONFIGURED,
    canUseDev: true,

    /** Current runtime mode. */
    getMode(): AuthMode {
      return _mode;
    },

    setMode(mode: AuthMode) {
      const next: AuthMode = !GOOGLE_CONFIGURED ? 'dev' : mode;
      _authMode.set(next);
      persist(null);
      set(null);
    },

    /** Dev-mode sign in: bind a chosen display name to a stable subject. */
    devSignIn(name: string) {
      if (_mode !== 'dev') {
        throw new Error('Switch mode to dev before using a local subject.');
      }
      const clean = name.trim() || 'guest';
      const sub = `dev:${slug(clean)}`;
      const user: AuthUser = { sub, name: clean, token: sub, dev: true };
      persist(user);
      set(user);
    },

    /** Complete direct Google sign-in by exchanging the GIS credential. */
    async signInWithGoogleCredential(credential: string) {
      if (!GOOGLE_CONFIGURED) {
        throw new Error('Google auth not configured in this build.');
      }
      if (_mode !== 'google') {
        throw new Error('Switch mode to Google before starting sign-in.');
      }
      const res = await exchangeGoogleToken(credential);
      const user: AuthUser = {
        sub: res.user.sub,
        name: res.user.name,
        email: res.user.email,
        picture: res.user.picture,
        token: res.token,
        expiresAt: res.expiresAt,
        refreshToken: res.refreshToken,
        dev: false
      };
      persist(user);
      set(user);
    },

    /**
     * Return a valid bearer token, refreshing silently when needed.
     *
     * Returns `null` when the user is signed out or the refresh token has
     * expired (in which case the store is cleared and the sign-in UI shows).
     * Pass `force` to refresh even if the current token still looks valid
     * (used to recover from a 401 caused by clock skew or secret rotation).
     */
    async getValidToken(force = false): Promise<string | null> {
      const user = current;
      if (!user) return null;
      if (user.dev) return user.token;
      if (force || accessTokenStale(user)) {
        return doRefresh(user);
      }
      return user.token;
    },

    /**
     * Restore a previously persisted Gaia session on app start.
     */
    async restore(): Promise<boolean> {
      const restored = load();
      if (!restored) return false;
      if (restored.dev) {
        set(restored);
        return true;
      }
      set(restored);
      if (accessTokenStale(restored)) {
        const token = await doRefresh(restored);
        return token !== null;
      }
      return true;
    },

    signOut() {
      persist(null);
      set(null);
    }
  };
}

function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export const auth = createAuth();
