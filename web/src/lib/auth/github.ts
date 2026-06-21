/**
 * GitHub OAuth helper for the browser (authorization-code flow).
 *
 * `startGithubSignIn` redirects the browser to GitHub's authorize endpoint.
 * After the user approves, GitHub redirects back to this app with `?code=&state=`.
 * `consumeGithubRedirect` validates the CSRF `state`, strips the OAuth params
 * from the URL, and returns the `{ code, redirectUri }` to exchange at
 * POST /v1/auth/github. The client secret never touches the browser — the code
 * is exchanged for a token server-side.
 */
import { browser } from '$app/environment';
import { GITHUB_CLIENT_ID } from '$lib/config';

/** GitHub authorize endpoint (authorization-code grant). */
const GITHUB_AUTHORIZE_URL = 'https://github.com/login/oauth/authorize';

/** sessionStorage key holding the random CSRF state between redirect legs. */
const STATE_KEY = 'gaia.github.oauth.state';

/** The OAuth scope we request: read-only access to the user's public profile. */
const GITHUB_SCOPE = 'read:user';

/** Result of consuming a GitHub redirect: the code plus the redirect URI used. */
export interface GithubRedirect {
  code: string;
  redirectUri: string;
}

/** Redirect URI for the current page, without any query string or hash. */
function currentRedirectUri(): string {
  return `${location.origin}${location.pathname}`;
}

/** Generate a random, URL-safe CSRF state value. */
function randomState(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Begin GitHub sign-in by redirecting to GitHub's authorize page.
 *
 * Stores a random `state` in sessionStorage so the redirect leg can defend
 * against CSRF. Throws when GitHub is not configured in this build.
 */
export function startGithubSignIn(): void {
  if (!browser) throw new Error('GitHub sign-in is only available in the browser.');
  if (!GITHUB_CLIENT_ID) {
    throw new Error('VITE_GITHUB_CLIENT_ID is not configured.');
  }

  const state = randomState();
  try {
    sessionStorage.setItem(STATE_KEY, state);
  } catch {
    // Without sessionStorage we cannot validate state safely; fail loudly.
    throw new Error('Unable to start GitHub sign-in: sessionStorage is unavailable.');
  }

  const params = new URLSearchParams({
    client_id: GITHUB_CLIENT_ID,
    redirect_uri: currentRedirectUri(),
    scope: GITHUB_SCOPE,
    state,
    allow_signup: 'false'
  });
  location.assign(`${GITHUB_AUTHORIZE_URL}?${params.toString()}`);
}

/**
 * Consume a GitHub OAuth redirect if the current URL carries `?code=&state=`.
 *
 * Validates the `state` against the value saved by {@link startGithubSignIn},
 * removes the OAuth params from the address bar, and returns the authorization
 * `code` plus the `redirectUri` that GitHub was given. Returns `null` when the
 * page was not reached via a GitHub redirect. Throws on a state mismatch
 * (possible CSRF) so the caller can surface an error instead of signing in.
 */
export function consumeGithubRedirect(): GithubRedirect | null {
  if (!browser) return null;

  const url = new URL(location.href);
  const code = url.searchParams.get('code');
  const state = url.searchParams.get('state');
  if (!code || !state) return null;

  let expected: string | null = null;
  try {
    expected = sessionStorage.getItem(STATE_KEY);
    sessionStorage.removeItem(STATE_KEY);
  } catch {
    expected = null;
  }

  // Strip the OAuth params so a refresh doesn't replay the exchange.
  url.searchParams.delete('code');
  url.searchParams.delete('state');
  history.replaceState(null, '', `${url.origin}${url.pathname}${url.search}${url.hash}`);

  if (!expected || expected !== state) {
    throw new Error('GitHub sign-in failed: state mismatch (possible CSRF).');
  }

  return { code, redirectUri: currentRedirectUri() };
}
