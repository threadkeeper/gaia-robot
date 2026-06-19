//! Google sign-in verification and in-memory session management.
//!
//! When `GOOGLE_CLIENT_ID` is set, the server runs in **live auth** mode:
//! the front end signs in with Google Identity Services and exchanges the
//! Google ID token for a Gaia session at `POST /v1/auth/google`. The session
//! consists of a short-lived access token (1 h) and a long-lived refresh token
//! (30 days), both opaque hex strings backed by an in-memory store.
//!
//! When `GOOGLE_CLIENT_ID` is *not* set the server falls back to **dev auth**:
//! the front end sends `Authorization: Bearer dev:<name>` and the server maps
//! the `<name>` straight to the `user_id` (no verification).
//!
//! Google ID tokens are verified by calling Google's `tokeninfo` endpoint over
//! HTTPS (via [`ureq`]). The returned `aud` claim must match the configured
//! client id.

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

// ---- Wire types ---------------------------------------------------------

/// User identity extracted from a verified Google ID token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// Google subject (unique, stable user id).
    pub sub: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub picture: Option<String>,
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
            sessions: Mutex::new(HashMap::new()),
            refresh_tokens: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }

    /// Read `GOOGLE_CLIENT_ID` from the environment and build an [`Auth`].
    pub fn from_env() -> Self {
        let client_id = crate::llm::value_from_env("GOOGLE_CLIENT_ID");
        if let Some(ref id) = client_id {
            println!("auth: live mode (Google client {id})");
        } else {
            println!("auth: dev mode (no GOOGLE_CLIENT_ID)");
        }
        Self::new(client_id)
    }

    /// Whether live Google sign-in is active.
    pub fn is_live(&self) -> bool {
        self.google_client_id.is_some()
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

    /// Resolve a bearer token to a `user_id`.
    ///
    /// In **dev mode** (no `GOOGLE_CLIENT_ID`): accepts `dev:<name>` tokens and
    /// maps `<name>` to the user id, falling back to the configured dev user.
    ///
    /// In **live mode**: looks up the token in the session store. If not found,
    /// still accepts `dev:` tokens as a convenience for local testing against
    /// the live backend.
    pub fn resolve_user_id(&self, bearer: Option<&str>) -> String {
        let token = match bearer {
            Some(t) => t.trim(),
            None => return crate::llm::dev_user_id(),
        };

        // Dev-auth tokens are always accepted (dev: prefix).
        if let Some(name) = token.strip_prefix("dev:") {
            return name.to_string();
        }

        // Try the session store.
        if let Some(user_id) = self.verify_access_token(token) {
            return user_id;
        }

        // In dev mode, treat any unknown token as the literal subject (backward
        // compat). In live mode, unknown tokens fall through to the dev user.
        if self.is_live() {
            crate::llm::dev_user_id()
        } else {
            token.to_string()
        }
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
    fn resolve_dev_token() {
        let auth = Auth::new(Some("client.apps.googleusercontent.com".to_string()));
        assert_eq!(auth.resolve_user_id(Some("dev:alice")), "alice");
    }

    #[test]
    fn resolve_session_token() {
        let auth = Auth::new(Some("client.apps.googleusercontent.com".to_string()));
        let exchange = auth.create_session(sample_user());
        assert_eq!(auth.resolve_user_id(Some(&exchange.token)), "google-123");
    }

    #[test]
    fn resolve_unknown_in_live_mode_falls_back() {
        let auth = Auth::new(Some("client.apps.googleusercontent.com".to_string()));
        // Unknown token in live mode → dev user id (not the token literal).
        let result = auth.resolve_user_id(Some("unknown-jwt-token"));
        // Should be the default dev user id, not "unknown-jwt-token".
        assert_ne!(result, "unknown-jwt-token");
    }

    #[test]
    fn resolve_no_bearer() {
        let auth = Auth::new(None);
        // No bearer → dev user id.
        let result = auth.resolve_user_id(None);
        assert!(!result.is_empty());
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
}
