/**
 * Authentication store.
 *
 * Sign-in is mandatory. The frontend authenticates with **Google** (collect a
 * Google ID token via GIS, exchange it at POST /v1/auth/google) or **GitHub**
 * (authorization-code redirect, exchange the code at POST /v1/auth/github). In
 * both cases the server returns an opaque Gaia session token that is sent as the
 * bearer on every API call. There is no dev/guest mode.
 */
import { browser } from '$app/environment';
import { exchangeGithubCode, exchangeGoogleToken, refreshSession } from '$lib/api/client';
import { GITHUB_CONFIGURED, GOOGLE_CONFIGURED } from '$lib/config';
import type { AuthUser } from '$lib/types';
import { writable } from 'svelte/store';

const STORAGE_KEY = 'gaia.auth.v1';
const MODE_KEY = 'gaia.auth.mode.v1';

/** Refresh the access token this many seconds before it actually expires. */
const REFRESH_SKEW_SECONDS = 60;

export type AuthMode = 'google' | 'github';

/** Pick a sensible default provider given what's configured in this build. */
function defaultMode(): AuthMode {
  return GOOGLE_CONFIGURED ? 'google' : 'github';
}

/** Coerce an arbitrary stored value into a usable, configured mode. */
function normalizeMode(value: string | null): AuthMode {
  if (value === 'github' && GITHUB_CONFIGURED) return 'github';
  if (value === 'google' && GOOGLE_CONFIGURED) return 'google';
  return defaultMode();
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
  try {
    return normalizeMode(localStorage.getItem(MODE_KEY));
  } catch {
    return defaultMode();
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
  _mode = normalizeMode(m);
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

  /** True when an access token is missing or within the refresh skew window. */
  function accessTokenStale(user: AuthUser): boolean {
    if (!user.expiresAt) return true;
    return user.expiresAt - REFRESH_SKEW_SECONDS <= Math.floor(Date.now() / 1000);
  }

  /** Build an `AuthUser` from a session exchange returned by the server. */
  function userFromExchange(
    res: Awaited<ReturnType<typeof exchangeGoogleToken>>,
    provider: AuthMode
  ): AuthUser {
    return {
      sub: res.user.sub,
      name: res.user.name,
      email: res.user.email,
      picture: res.user.picture,
      githubLogin: res.user.githubLogin,
      provider,
      token: res.token,
      expiresAt: res.expiresAt,
      refreshToken: res.refreshToken
    };
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
          refreshToken: res.refreshToken
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
    canUseGithub: GITHUB_CONFIGURED,

    /** Current runtime mode. */
    getMode(): AuthMode {
      return _mode;
    },

    setMode(mode: AuthMode) {
      _authMode.set(normalizeMode(mode));
    },

    /** Complete direct Google sign-in by exchanging the GIS credential. */
    async signInWithGoogleCredential(credential: string) {
      if (!GOOGLE_CONFIGURED) {
        throw new Error('Google auth not configured in this build.');
      }
      const res = await exchangeGoogleToken(credential);
      const user = userFromExchange(res, 'google');
      persist(user);
      set(user);
    },

    /** Complete GitHub sign-in by exchanging an authorization code. */
    async signInWithGithubCode(code: string, redirectUri?: string) {
      if (!GITHUB_CONFIGURED) {
        throw new Error('GitHub auth not configured in this build.');
      }
      const res = await exchangeGithubCode(code, redirectUri);
      const user = userFromExchange(res, 'github');
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

export const auth = createAuth();
