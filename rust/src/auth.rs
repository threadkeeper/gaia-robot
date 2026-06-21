//! Google sign-in verification and in-memory session management.
//!
//! When `GOOGLE_CLIENT_ID` is set, the server runs in **live auth** mode:
//! the front end signs in with Google Identity Services and exchanges the
//! Google ID token for a Gaia session at `POST /v1/auth/google`. The session
//! consists of a short-lived access token (1 h) and a long-lived refresh token
//! (30 days), both opaque hex strings backed by an in-memory store.
//!
//! Sign-in is **mandatory**: users authenticate with Google or GitHub and the
//! server mints a session for them. The session consists of a short-lived
//! access token (1 h) and a long-lived refresh token (30 days), both opaque hex
//! strings backed by an in-memory store. Every protected request must carry a
//! valid access token — there is no dev/guest fallback, and `dev:` or unknown
//! tokens are rejected.
//!
//! Google ID tokens are verified by calling Google's `tokeninfo` endpoint over
//! HTTPS (via [`ureq`]). The returned `aud` claim must match the configured
//! client id.
//!
//! **GitHub sign-in** is offered as an additional option. Because GitHub has no
//! browser-verifiable ID token, it uses the OAuth *authorization-code* flow: the
//! front end redirects to GitHub, GitHub redirects back with a `?code=`, and the
//! front end posts that code to `POST /v1/auth/github`. The server exchanges the
//! code (plus its client secret) for a GitHub access token and fetches the
//! user's profile, then mints the same kind of Gaia session as Google does.
//! GitHub sign-in is active when both `GITHUB_CLIENT_ID` and
//! `GITHUB_CLIENT_SECRET` are set.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Access-token lifetime: 1 hour.
const ACCESS_TTL_SECS: u64 = 3600;
/// Refresh-token lifetime: 30 days.
const REFRESH_TTL_SECS: u64 = 30 * 24 * 3600;
/// Google's public token-info endpoint.
const GOOGLE_TOKENINFO: &str = "https://oauth2.googleapis.com/tokeninfo";
/// GitHub OAuth access-token exchange endpoint (authorization-code flow).
const GITHUB_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
/// GitHub authenticated-user profile endpoint.
const GITHUB_USER_URL: &str = "https://api.github.com/user";
/// GitHub requires a `User-Agent` on every API request; identify this app.
const GITHUB_USER_AGENT: &str = "gaia-robot";

// ---- Wire types ---------------------------------------------------------

/// User identity extracted from a verified sign-in (Google ID token or GitHub
/// profile).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// Stable, unique subject and the user's wing id. For Google this is the
    /// Google `sub`; for GitHub it is `github:<numeric-id>`.
    pub sub: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub picture: Option<String>,
    /// GitHub login handle, present only for GitHub sign-in. Surfaced to the
    /// front end as `githubLogin` for display.
    #[serde(rename = "githubLogin", skip_serializing_if = "Option::is_none")]
    pub github_login: Option<String>,
}

/// Response returned to the front end from `/v1/auth/google` and
/// `/v1/auth/refresh`, matching the `GoogleAuthExchange` TypeScript interface.
#[derive(Debug, Serialize)]
pub struct AuthExchange {
    /// Short-lived bearer access token.
    pub token: String,
    /// Unix seconds when `token` expires.
    #[serde(rename = "expiresAt")]
    pub expires_at: u64,
    /// Long-lived refresh token for silent renewal.
    #[serde(rename = "refreshToken")]
    pub refresh_token: String,
    /// Authenticated user profile.
    pub user: UserInfo,
}

// ---- Internal session store ---------------------------------------------

/// An active access-token session.
struct Session {
    user_id: String,
    /// Kept for potential future use (e.g. returning user info on token
    /// introspection), but currently only `user_id` is read.
    #[allow(dead_code)]
    user_info: UserInfo,
    expires_at: u64,
}

/// A stored refresh token and the identity it belongs to.
struct RefreshEntry {
    user_info: UserInfo,
    expires_at: u64,
}

// ---- Auth manager -------------------------------------------------------

/// Google sign-in verifier and in-memory session store.
///
/// Thread-safe: the session maps are behind [`Mutex`]es and the token counter
/// is atomic, so the per-connection threads in [`crate::server::Server`] can
/// call into `Auth` concurrently.
pub struct Auth {
    /// When `Some`, live-auth is active and the `aud` claim of every Google
    /// token must match this value. When `None`, only dev-auth is available.
    google_client_id: Option<String>,
    /// GitHub OAuth client id. When this and [`Self::github_client_secret`] are
    /// both `Some`, GitHub sign-in is available.
    github_client_id: Option<String>,
    /// GitHub OAuth client secret, used server-side to exchange the code.
    github_client_secret: Option<String>,
    /// Access-token → session map.
    sessions: Mutex<HashMap<String, Session>>,
    /// Refresh-token → user-info map.
    refresh_tokens: Mutex<HashMap<String, RefreshEntry>>,
    /// Monotonic counter mixed into token generation for uniqueness.
    counter: AtomicU64,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Auth")
            .field("live", &self.is_live())
            .finish()
    }
}

impl Auth {
    /// Create an auth manager.
    ///
    /// Pass `Some(client_id)` to enable Google sign-in verification; pass
    /// `None` for dev-auth-only mode.
    pub fn new(google_client_id: Option<String>) -> Self {
        Self {
            google_client_id,
            github_client_id: None,
            github_client_secret: None,
            sessions: Mutex::new(HashMap::new()),
            refresh_tokens: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }

    /// Enable GitHub sign-in by attaching an OAuth client id and secret.
    ///
    /// Returns `self` so it can be chained after [`Self::new`]. GitHub sign-in
    /// only activates when *both* values are non-empty.
    pub fn with_github_oauth(
        mut self,
        client_id: Option<String>,
        client_secret: Option<String>,
    ) -> Self {
        // Treat blank env values as "unset" so a stray empty var never
        // half-enables the flow.
        self.github_client_id = client_id.filter(|v| !v.trim().is_empty());
        self.github_client_secret = client_secret.filter(|v| !v.trim().is_empty());
        self
    }

    /// Read auth configuration from the environment and build an [`Auth`].
    ///
    /// `GOOGLE_CLIENT_ID` enables Google sign-in; `GITHUB_CLIENT_ID` +
    /// `GITHUB_CLIENT_SECRET` enable GitHub sign-in. Either, both, or neither
    /// may be configured.
    pub fn from_env() -> Self {
        let client_id = crate::llm::value_from_env("GOOGLE_CLIENT_ID");
        if let Some(ref id) = client_id {
            println!("auth: live mode (Google client {id})");
        } else {
            println!("auth: dev mode (no GOOGLE_CLIENT_ID)");
        }

        let github_id = crate::llm::value_from_env("GITHUB_CLIENT_ID");
        let github_secret = crate::llm::value_from_env("GITHUB_CLIENT_SECRET");
        let auth = Self::new(client_id).with_github_oauth(github_id, github_secret);
        if auth.is_github_live() {
            println!("auth: GitHub sign-in enabled");
        }
        auth
    }

    /// Whether live Google sign-in is active.
    pub fn is_live(&self) -> bool {
        self.google_client_id.is_some()
    }

    /// Whether GitHub sign-in is active (both client id and secret configured).
    pub fn is_github_live(&self) -> bool {
        self.github_client_id.is_some() && self.github_client_secret.is_some()
    }

    // ---- Google token verification --------------------------------------

    /// Verify a Google ID token by calling Google's `tokeninfo` endpoint.
    ///
    /// Returns the extracted [`UserInfo`] on success, or a human-readable
    /// error string on failure.
    pub fn verify_google_token(&self, id_token: &str) -> Result<UserInfo, String> {
        let client_id = self
            .google_client_id
            .as_deref()
            .ok_or("google auth is not configured (GOOGLE_CLIENT_ID not set)")?;

        // Call Google's tokeninfo endpoint to verify the token. This is the
        // recommended server-side verification approach when you don't want to
        // implement local JWKS validation.
        let url = format!("{GOOGLE_TOKENINFO}?id_token={id_token}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| format!("google tokeninfo request failed: {e}"))?;

        let body_str = resp
            .into_string()
            .map_err(|e| format!("failed to read tokeninfo response: {e}"))?;

        let body: serde_json::Value = serde_json::from_str(&body_str)
            .map_err(|e| format!("failed to parse tokeninfo response: {e}"))?;

        // The `aud` claim must match our configured client id.
        let aud = body["aud"].as_str().unwrap_or_default();
        if aud != client_id {
            return Err(format!(
                "token audience mismatch: expected {client_id}, got {aud}"
            ));
        }

        Ok(UserInfo {
            sub: body["sub"].as_str().ok_or("missing sub claim")?.to_string(),
            name: body["name"].as_str().map(str::to_string),
            email: body["email"].as_str().map(str::to_string),
            picture: body["picture"].as_str().map(str::to_string),
            github_login: None,
        })
    }

    // ---- GitHub OAuth code exchange -------------------------------------

    /// Exchange a GitHub OAuth `code` for the authenticated user's [`UserInfo`].
    ///
    /// Runs the server side of GitHub's authorization-code flow: POST the code
    /// (with the client id/secret) to GitHub's token endpoint, then call the
    /// `/user` API with the returned access token. `redirect_uri` must match the
    /// value the browser used to start the flow (GitHub validates it when it was
    /// supplied); pass `None` to omit it.
    ///
    /// Returns a human-readable error string on any configuration or HTTP
    /// failure, failing closed so a bad code never yields a session.
    pub fn exchange_github_code(
        &self,
        code: &str,
        redirect_uri: Option<&str>,
    ) -> Result<UserInfo, String> {
        let client_id = self
            .github_client_id
            .as_deref()
            .ok_or("github auth is not configured (GITHUB_CLIENT_ID not set)")?;
        let client_secret = self
            .github_client_secret
            .as_deref()
            .ok_or("github auth is not configured (GITHUB_CLIENT_SECRET not set)")?;
        if code.trim().is_empty() {
            return Err("missing github code".to_string());
        }

        // 1. Exchange the authorization code for a GitHub access token.
        let mut request = serde_json::json!({
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
        });
        if let Some(uri) = redirect_uri.filter(|u| !u.is_empty()) {
            request["redirect_uri"] = serde_json::Value::String(uri.to_string());
        }
        let payload = serde_json::to_vec(&request)
            .map_err(|e| format!("failed to serialize github token request: {e}"))?;

        let token_resp = ureq::post(GITHUB_TOKEN_URL)
            .set("Accept", "application/json")
            .set("Content-Type", "application/json")
            .set("User-Agent", GITHUB_USER_AGENT)
            .send_bytes(&payload)
            .map_err(|e| format!("github token request failed: {e}"))?;
        let token_body = token_resp
            .into_string()
            .map_err(|e| format!("failed to read github token response: {e}"))?;
        let token_json: serde_json::Value = serde_json::from_str(&token_body)
            .map_err(|e| format!("failed to parse github token response: {e}"))?;

        // GitHub returns HTTP 200 with an `error` field on bad/expired codes, so
        // a missing access token is the real failure signal here.
        let access_token = token_json["access_token"].as_str().unwrap_or_default();
        if access_token.is_empty() {
            let reason = token_json["error"].as_str().unwrap_or("no access_token");
            return Err(format!("github token exchange failed: {reason}"));
        }

        // 2. Fetch the authenticated user's profile with the access token.
        let user_resp = ureq::get(GITHUB_USER_URL)
            .set("Authorization", &format!("Bearer {access_token}"))
            .set("Accept", "application/vnd.github+json")
            .set("User-Agent", GITHUB_USER_AGENT)
            .call()
            .map_err(|e| format!("github user request failed: {e}"))?;
        let user_body = user_resp
            .into_string()
            .map_err(|e| format!("failed to read github user response: {e}"))?;
        let user: serde_json::Value = serde_json::from_str(&user_body)
            .map_err(|e| format!("failed to parse github user response: {e}"))?;

        let login = user["login"].as_str().unwrap_or_default();
        // Require both a login and a numeric id; the id is the stable subject.
        let id = match user["id"].as_i64() {
            Some(id) if !login.is_empty() => id,
            _ => return Err("github profile missing login/id".to_string()),
        };

        // Empty profile strings (GitHub uses null/"" for hidden email, no name)
        // are normalized to `None`; the login is a sensible display fallback.
        let non_empty = |key: &str| {
            user[key]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        Ok(UserInfo {
            sub: format!("github:{id}"),
            name: non_empty("name").or_else(|| Some(login.to_string())),
            email: non_empty("email"),
            picture: non_empty("avatar_url"),
            github_login: Some(login.to_string()),
        })
    }

    // ---- Session management ---------------------------------------------

    /// Create a new session for a verified user and return the exchange
    /// payload the front end expects.
    pub fn create_session(&self, info: UserInfo) -> AuthExchange {
        let now = now_secs();
        let access_token = self.generate_token();
        let refresh_token = self.generate_token();

        let session = Session {
            user_id: info.sub.clone(),
            user_info: info.clone(),
            expires_at: now + ACCESS_TTL_SECS,
        };
        self.sessions
            .lock()
            .expect("session lock poisoned")
            .insert(access_token.clone(), session);

        let refresh = RefreshEntry {
            user_info: info.clone(),
            expires_at: now + REFRESH_TTL_SECS,
        };
        self.refresh_tokens
            .lock()
            .expect("refresh lock poisoned")
            .insert(refresh_token.clone(), refresh);

        AuthExchange {
            token: access_token,
            expires_at: now + ACCESS_TTL_SECS,
            refresh_token,
            user: info,
        }
    }

    /// Verify an access token and return the `user_id` it maps to, or `None`
    /// if the token is unknown or expired.
    pub fn verify_access_token(&self, token: &str) -> Option<String> {
        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        if let Some(session) = sessions.get(token) {
            if session.expires_at > now_secs() {
                return Some(session.user_id.clone());
            }
            // Expired — remove it.
            sessions.remove(token);
        }
        None
    }

    /// Exchange a refresh token for a fresh session.
    ///
    /// The old refresh token is consumed (single use) and a new one is issued.
    pub fn refresh(&self, refresh_token: &str) -> Option<AuthExchange> {
        let entry = self
            .refresh_tokens
            .lock()
            .expect("refresh lock poisoned")
            .remove(refresh_token)?;

        if entry.expires_at <= now_secs() {
            return None; // expired
        }

        Some(self.create_session(entry.user_info))
    }

    /// Authenticate a bearer token, returning the `user_id` of a **valid
    /// session** or `None`.
    ///
    /// This is the enforcement path for protected HTTP routes: it accepts only
    /// tokens minted by [`Self::create_session`] after a successful Google or
    /// GitHub sign-in. Missing, malformed, unknown, or expired tokens all yield
    /// `None`, so unauthenticated callers are rejected — there is deliberately no
    /// `dev:` token or guest fallback.
    pub fn authenticate(&self, bearer: Option<&str>) -> Option<String> {
        let token = bearer?.trim();
        if token.is_empty() {
            return None;
        }
        self.verify_access_token(token)
    }

    // ---- Token generation -----------------------------------------------

    /// Generate a 32-hex-char (128-bit) opaque token.
    ///
    /// Uses [`RandomState`] (OS-seeded SipHash) mixed with a monotonic counter
    /// and the current timestamp. Not a CSPRNG, but adequate for session tokens
    /// in a single-container app behind HTTPS.
    fn generate_token(&self) -> String {
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let h1 = {
            let mut h = RandomState::new().build_hasher();
            count.hash(&mut h);
            nanos.hash(&mut h);
            std::process::id().hash(&mut h);
            h.finish()
        };

        let h2 = {
            let mut h = RandomState::new().build_hasher();
            (count.wrapping_add(1)).hash(&mut h);
            nanos.hash(&mut h);
            h.finish()
        };

        format!("{h1:016x}{h2:016x}")
    }
}

/// Current wall-clock time as Unix seconds.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_user() -> UserInfo {
        UserInfo {
            sub: "google-123".to_string(),
            name: Some("Alice".to_string()),
            email: Some("alice@example.com".to_string()),
            picture: None,
            github_login: None,
        }
    }

    #[test]
    fn dev_mode_when_no_client_id() {
        let auth = Auth::new(None);
        assert!(!auth.is_live());
    }

    #[test]
    fn live_mode_when_client_id_set() {
        let auth = Auth::new(Some("my-client.apps.googleusercontent.com".to_string()));
        assert!(auth.is_live());
    }

    #[test]
    fn create_and_verify_session() {
        let auth = Auth::new(None);
        let exchange = auth.create_session(sample_user());

        // The access token should resolve to the user's sub.
        assert_eq!(
            auth.verify_access_token(&exchange.token),
            Some("google-123".to_string())
        );
        // The exchange fields are populated.
        assert_eq!(exchange.user.sub, "google-123");
        assert!(!exchange.token.is_empty());
        assert!(!exchange.refresh_token.is_empty());
        assert!(exchange.expires_at > 0);
    }

    #[test]
    fn refresh_rotates_tokens() {
        let auth = Auth::new(None);
        let first = auth.create_session(sample_user());
        let refresh_tok = first.refresh_token.clone();

        // Refresh should succeed and give new tokens.
        let second = auth.refresh(&refresh_tok).expect("refresh should succeed");
        assert_ne!(second.token, first.token);
        assert_ne!(second.refresh_token, first.refresh_token);
        assert_eq!(second.user.sub, "google-123");

        // The old refresh token is consumed — second use fails.
        assert!(auth.refresh(&refresh_tok).is_none());
    }

    #[test]
    fn unknown_token_returns_none() {
        let auth = Auth::new(None);
        assert!(auth.verify_access_token("bogus").is_none());
    }

    #[test]
    fn authenticate_rejects_dev_token() {
        // Dev tokens are no longer accepted: sign-in is mandatory.
        let auth = Auth::new(Some("client.apps.googleusercontent.com".to_string()));
        assert_eq!(auth.authenticate(Some("dev:alice")), None);
    }

    #[test]
    fn authenticate_accepts_session_token() {
        let auth = Auth::new(Some("client.apps.googleusercontent.com".to_string()));
        let exchange = auth.create_session(sample_user());
        assert_eq!(
            auth.authenticate(Some(&exchange.token)),
            Some("google-123".to_string())
        );
    }

    #[test]
    fn authenticate_rejects_unknown_token() {
        let auth = Auth::new(Some("client.apps.googleusercontent.com".to_string()));
        assert_eq!(auth.authenticate(Some("unknown-jwt-token")), None);
    }

    #[test]
    fn authenticate_rejects_missing_bearer() {
        let auth = Auth::new(None);
        assert_eq!(auth.authenticate(None), None);
        assert_eq!(auth.authenticate(Some("   ")), None);
    }

    #[test]
    fn generated_tokens_are_unique() {
        let auth = Auth::new(None);
        let t1 = auth.generate_token();
        let t2 = auth.generate_token();
        assert_ne!(t1, t2);
        assert_eq!(t1.len(), 32); // 16 hex chars × 2
    }

    #[test]
    fn verify_google_fails_without_client_id() {
        let auth = Auth::new(None);
        let err = auth.verify_google_token("fake-token").unwrap_err();
        assert!(err.contains("not configured"));
    }

    #[test]
    fn github_disabled_by_default() {
        let auth = Auth::new(None);
        assert!(!auth.is_github_live());
    }

    #[test]
    fn github_enabled_when_id_and_secret_set() {
        let auth =
            Auth::new(None).with_github_oauth(Some("id".to_string()), Some("secret".to_string()));
        assert!(auth.is_github_live());
    }

    #[test]
    fn github_disabled_when_secret_missing() {
        // An id without a secret must not half-enable the flow.
        let auth = Auth::new(None).with_github_oauth(Some("id".to_string()), None);
        assert!(!auth.is_github_live());
    }

    #[test]
    fn github_disabled_when_values_blank() {
        // Blank env values are treated as unset.
        let auth = Auth::new(None).with_github_oauth(Some("  ".to_string()), Some("".to_string()));
        assert!(!auth.is_github_live());
    }

    #[test]
    fn exchange_github_fails_without_config() {
        let auth = Auth::new(None);
        let err = auth.exchange_github_code("some-code", None).unwrap_err();
        assert!(err.contains("not configured"));
    }

    #[test]
    fn exchange_github_rejects_empty_code() {
        let auth =
            Auth::new(None).with_github_oauth(Some("id".to_string()), Some("secret".to_string()));
        let err = auth.exchange_github_code("   ", None).unwrap_err();
        assert!(err.contains("missing github code"));
    }

    #[test]
    fn github_session_uses_subject_as_user_id() {
        // A GitHub-derived identity flows through the same session machinery.
        let auth = Auth::new(None);
        let info = UserInfo {
            sub: "github:42".to_string(),
            name: Some("octocat".to_string()),
            email: None,
            picture: None,
            github_login: Some("octocat".to_string()),
        };
        let exchange = auth.create_session(info);
        assert_eq!(
            auth.authenticate(Some(&exchange.token)),
            Some("github:42".to_string())
        );
        assert_eq!(exchange.user.github_login.as_deref(), Some("octocat"));
    }
}
