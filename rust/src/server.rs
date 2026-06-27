//! The [`Server`] type: Gaia's HTTP + WebSocket backend.
//!
//! The server exposes exactly the endpoints the PWA front end calls (after the
//! deprecated SSE streaming endpoint was removed):
//!
//! | Method | Path | Purpose |
//! |---|---|---|
//! | `GET`  | `/healthz` | liveness (cheap; proves the process is up) |
//! | `GET`  | `/readyz` | readiness (deep; probes every configured dependency) |
//! | `POST` | `/v1/conversations/{id}/messages` | run one turn, return the reply |
//! | `GET`  | `/v1/ws/{id}` (Upgrade) | run one turn, stream the reply over WS |
//! | `POST` | `/v1/auth/google` | exchange a Google ID token for a session |
//! | `POST` | `/v1/auth/github` | exchange a GitHub OAuth code for a session |
//! | `POST` | `/v1/auth/refresh` | exchange a refresh token for a fresh session |
//!
//! When `GAIA_WEB_DIR` is set it also serves the bundled PWA front end: any
//! non-API `GET` returns a built static file, falling back to `index.html` for
//! client-side routes. That lets a single Container App host both the API and the
//! installable web app on one origin (no CORS, no separate Static Web App).
//!
//! It is a small, blocking, thread-per-connection server built directly on
//! [`std::net`] — no async runtime and no web framework — matching the project's
//! preference for the standard library and its hand-rolled protocol code
//! ([`crate::cosmos`]). Each connection is parsed with
//! [`crate::http_request::HttpRequest`], dispatched here, and answered with a
//! [`crate::http_response::HttpResponse`] (or upgraded to a WebSocket).
//!
//! **Auth model:** sign-in is mandatory. The front end signs in with Google or
//! GitHub, exchanges the result for a Gaia session at `POST /v1/auth/{google,
//! github}`, and then sends the session access token as `Authorization: Bearer
//! <token>` (or a `{token}` WebSocket hello). Protected routes reject any request
//! that does not carry a valid session token with `401` — there is no dev/guest
//! fallback.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use crate::auth::Auth;
use crate::engine::Engine;
use crate::http_request::HttpRequest;
use crate::http_response::HttpResponse;
use crate::static_files::StaticSite;
use crate::websocket::{self, Message};

/// How long a single connection may stay idle before we give up on it. Keeps
/// dropped/abandoned sockets from tying up a thread forever.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(120);

/// Gaia's backend HTTP/WebSocket server.
///
/// Holds a shared [`Engine`] (the turn runner) and [`Auth`] (session manager)
/// behind [`Arc`]s so every connection thread can run turns concurrently. When a
/// [`StaticSite`] is configured it also serves the bundled PWA front end from the
/// same origin, so one Container App hosts both the API and the web app.
#[derive(Debug, Clone)]
pub struct Server {
    engine: Arc<Engine>,
    auth: Arc<Auth>,
    /// The bundled PWA, served for non-API GET requests when present.
    site: Option<Arc<StaticSite>>,
}

impl Server {
    /// Build a server around an already-configured engine and auth manager.
    ///
    /// `site` is the optional static front end (from `GAIA_WEB_DIR`); pass `None`
    /// to run API-only.
    pub fn new(engine: Engine, auth: Auth, site: Option<StaticSite>) -> Self {
        Server {
            engine: Arc::new(engine),
            auth: Arc::new(auth),
            site: site.map(Arc::new),
        }
    }

    /// Bind to `addr` and serve connections forever (one thread per connection).
    ///
    /// Returns an error only if the initial bind fails; per-connection errors are
    /// logged and isolated so one bad client never takes the server down.
    pub fn serve(&self, addr: &str) -> std::io::Result<()> {
        let listener = TcpListener::bind(addr)?;
        let local = listener.local_addr()?;
        println!("Gaia backend listening on http://{local}");
        println!("  engine: {}", self.engine.describe());
        println!("  routes: GET /healthz, POST /v1/conversations/{{id}}/messages, GET /v1/ws/{{id}} (WebSocket)");

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let engine = Arc::clone(&self.engine);
                    let auth = Arc::clone(&self.auth);
                    let site = self.site.clone();
                    // One thread per connection keeps the model blocking-simple.
                    std::thread::spawn(move || {
                        if let Err(err) = handle_connection(stream, &engine, &auth, site.as_deref())
                        {
                            // Connection-level failures are expected (clients drop,
                            // time out, send garbage); log and move on.
                            eprintln!("connection error: {err}");
                        }
                    });
                }
                Err(err) => eprintln!("accept error: {err}"),
            }
        }
        Ok(())
    }
}

/// Resolve the listen address from the environment, or `None` to run the console.
///
/// The server starts when `GAIA_HTTP_ADDR` is set (e.g. `0.0.0.0:8080`) or
/// `GAIA_HTTP_PORT` is set (bound on `0.0.0.0:<port>`). Returning `None` leaves
/// the program in its default interactive console mode, so existing behaviour and
/// the CLI tests are unchanged.
pub fn http_addr_from_env() -> Option<String> {
    if let Some(addr) = crate::llm::value_from_env("GAIA_HTTP_ADDR") {
        return Some(addr);
    }
    crate::llm::value_from_env("GAIA_HTTP_PORT").map(|port| format!("0.0.0.0:{port}"))
}

/// Handle one client connection: parse a request, dispatch it, write the reply.
fn handle_connection(
    stream: TcpStream,
    engine: &Engine,
    auth: &Auth,
    site: Option<&StaticSite>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(CONNECTION_TIMEOUT))?;
    stream.set_write_timeout(Some(CONNECTION_TIMEOUT))?;

    // Read from a buffered clone; write responses to the original stream.
    let read_half = stream.try_clone()?;
    let mut reader = BufReader::new(read_half);
    let mut writer = stream;

    let request = match HttpRequest::read(&mut reader)? {
        Some(request) => request,
        None => return Ok(()), // client closed without sending anything
    };

    // CORS preflight: answer OPTIONS immediately with the allow headers.
    if request.method == "OPTIONS" {
        return cors(HttpResponse::empty(204, "No Content")).write_to(&mut writer);
    }

    // WebSocket upgrade for the chat stream.
    if request.is_websocket_upgrade() && request.path.starts_with("/v1/ws/") {
        return handle_websocket(&mut reader, &mut writer, &request, engine, auth);
    }

    let response = cors(route(&request, engine, auth, site));
    // A concise access-log line aids local debugging of the front end wiring.
    println!(
        "{} {} -> {}",
        request.method,
        request.path,
        response.status()
    );
    response.write_to(&mut writer)
}

/// Map a parsed request to a response (non-WebSocket routes).
fn route(
    request: &HttpRequest,
    engine: &Engine,
    auth: &Auth,
    site: Option<&StaticSite>,
) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        // Liveness probe (no auth): cheap, proves the process is up. Used by the
        // CD smoke test and any ingress health probe, so it stays trivial.
        ("GET", "/healthz") => HttpResponse::text(200, "OK", "ok"),

        // Readiness probe (no auth): actively checks every configured dependency.
        ("GET", "/readyz") => handle_readyz(engine),

        // Run one turn and return the reply.
        ("POST", path) if is_messages_path(path) => handle_message(request, engine, auth),

        // Exchange a Google ID token for a Gaia session.
        ("POST", "/v1/auth/google") => handle_auth_google(request, auth),

        // Exchange a GitHub OAuth code for a Gaia session.
        ("POST", "/v1/auth/github") => handle_auth_github(request, auth),

        // Refresh an expired access token.
        ("POST", "/v1/auth/refresh") => handle_auth_refresh(request, auth),

        // Static front end: serve the bundled PWA for any non-API GET. API paths
        // (/v1/...) are intentionally excluded so a missing API route still 404s
        // as JSON rather than silently returning the app shell.
        ("GET", path) if !path.starts_with("/v1/") => match site {
            Some(site) => site.response_for(path),
            None => HttpResponse::with_status_json(404, "Not Found", r#"{"error":"not found"}"#),
        },

        // Unknown route.
        _ => HttpResponse::with_status_json(404, "Not Found", r#"{"error":"not found"}"#),
    }
}

/// Handle `GET /readyz`: probe every configured dependency and report.
///
/// Returns the [`HealthReport`](crate::health::HealthReport) as JSON with HTTP
/// 200 when the service is ready (no configured dependency failed) or HTTP 503
/// when any configured dependency is unreachable, so orchestrators and humans
/// can distinguish "up but degraded" from "fully ready".
fn handle_readyz(engine: &Engine) -> HttpResponse {
    let report = engine.check_health();
    let body = match serde_json::to_vec(&report) {
        Ok(bytes) => bytes,
        Err(e) => {
            return HttpResponse::with_status_json(
                500,
                "Internal Server Error",
                format!(r#"{{"error":"failed to serialize readiness report: {e}"}}"#),
            )
        }
    };
    if report.ready {
        HttpResponse::with_status_json(200, "OK", body)
    } else {
        HttpResponse::with_status_json(503, "Service Unavailable", body)
    }
}

/// Handle `POST /v1/conversations/{id}/messages`: parse the body, run the turn.
fn handle_message(request: &HttpRequest, engine: &Engine, auth: &Auth) -> HttpResponse {
    let text = match parse_message_body(&request.body) {
        Some(text) => text,
        None => {
            return HttpResponse::with_status_json(
                400,
                "Bad Request",
                r#"{"error":"expected JSON body {\"text\":\"...\"}"}"}"#,
            )
        }
    };

    let user_id = match auth.authenticate(request.bearer_token()) {
        Some(id) => id,
        None => {
            return HttpResponse::with_status_json(
                401,
                "Unauthorized",
                r#"{"error":"authentication required"}"#,
            )
        }
    };
    let result = engine.run_turn(&user_id, &text);

    match serde_json::to_vec(&result) {
        Ok(json) => HttpResponse::json(json),
        Err(err) => HttpResponse::with_status_json(
            500,
            "Internal Server Error",
            format!(r#"{{"error":"failed to serialize reply: {err}"}}"#),
        ),
    }
}

/// Upgrade the connection to a WebSocket and stream one turn's reply.
///
/// Protocol (mirrors the front end's `streamWS`): the client sends a hello
/// `{"token":"<bearer>"}` then `{"text":"<message>"}`. We reply with a single
/// `{"type":"token","token":"<full reply>"}` frame followed by
/// `{"type":"done","result":<ReplyResult>}`, then close.
fn handle_websocket(
    reader: &mut BufReader<TcpStream>,
    writer: &mut TcpStream,
    request: &HttpRequest,
    engine: &Engine,
    auth: &Auth,
) -> std::io::Result<()> {
    // Complete the RFC 6455 opening handshake.
    let key = request.header("sec-websocket-key").unwrap_or_default();
    let accept = websocket::accept_key(key);
    let handshake = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\r\n"
    );
    writer.write_all(handshake.as_bytes())?;
    writer.flush()?;

    // Authenticate from the request's bearer (if any); the client's `{token}`
    // hello refines it before the turn runs. `None` means "not yet
    // authenticated" — a turn is refused until a valid session token arrives.
    let mut user_id = auth.authenticate(request.bearer_token());

    while let Some(message) = websocket::read_message(reader)? {
        match message {
            Message::Text(text) => {
                let value: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(_) => continue, // ignore non-JSON frames
                };

                // A hello frame carries the auth token; map it to the user id.
                if let Some(token) = value.get("token").and_then(|t| t.as_str()) {
                    user_id = auth.authenticate(Some(token));
                }

                // A text frame carries the user's message; run the turn only
                // for an authenticated session, otherwise reject and close.
                if let Some(input) = value.get("text").and_then(|t| t.as_str()) {
                    match &user_id {
                        Some(id) => {
                            let result = engine.run_turn(id, input);
                            stream_turn(writer, &result)?;
                        }
                        None => {
                            websocket::write_text(
                                writer,
                                r#"{"type":"error","error":"authentication required"}"#,
                            )?;
                        }
                    }
                    let _ = websocket::write_close(writer);
                    break;
                }
            }
            Message::Ping(payload) => websocket::write_pong(writer, &payload)?,
            Message::Close => break,
            // Binary / pong frames are not part of our protocol; ignore them.
            Message::Binary(_) | Message::Pong(_) => {}
        }
    }

    Ok(())
}

/// Send a completed turn over the WebSocket as a `token` frame then a `done`
/// frame, matching the front end's expected `{type:'token'|'done'}` envelope.
fn stream_turn(writer: &mut TcpStream, result: &crate::engine::TurnResult) -> std::io::Result<()> {
    // One token frame carrying the whole reply (the model is not itself
    // streaming), so the bubble fills before the metadata arrives.
    let token_frame = TokenFrame {
        kind: "token",
        token: &result.reply,
    };
    websocket::write_text(writer, &to_json(&token_frame))?;

    // The done frame carries the full structured result for the metadata panel.
    let done_frame = DoneFrame {
        kind: "done",
        result,
    };
    websocket::write_text(writer, &to_json(&done_frame))
}

/// A `{type:'token', token:'...'}` WebSocket frame.
#[derive(serde::Serialize)]
struct TokenFrame<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    token: &'a str,
}

/// A `{type:'done', result:<ReplyResult>}` WebSocket frame.
#[derive(serde::Serialize)]
struct DoneFrame<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    result: &'a crate::engine::TurnResult,
}

/// Serialize a frame to a JSON string, degrading to an error frame on the
/// (practically impossible) serialization failure rather than panicking.
fn to_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|_| r#"{"type":"error","error":"serialization failed"}"#.to_string())
}

/// Wrap a response with permissive CORS headers.
///
/// The front end authenticates with the `Authorization` header (not cookies), so
/// a wildcard origin is safe here and keeps cross-origin dev setups (a direct
/// `VITE_API_BASE`) working without per-origin configuration.
fn cors(response: HttpResponse) -> HttpResponse {
    response
        .with_header("Access-Control-Allow-Origin", "*")
        .with_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        .with_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type",
        )
}

/// Handle `POST /v1/auth/google`: verify a Google ID token, create a session.
fn handle_auth_google(request: &HttpRequest, auth: &Auth) -> HttpResponse {
    let id_token = match parse_json_field(&request.body, "idToken") {
        Some(t) => t,
        None => {
            return HttpResponse::with_status_json(
                400,
                "Bad Request",
                "{\"error\":\"expected JSON body with idToken\"}",
            )
        }
    };

    match auth.verify_google_token(&id_token) {
        Ok(info) => {
            let exchange = auth.create_session(info);
            match serde_json::to_vec(&exchange) {
                Ok(json) => HttpResponse::json(json),
                Err(err) => HttpResponse::with_status_json(
                    500,
                    "Internal Server Error",
                    format!("{{\"error\":\"serialize: {err}\"}}"),
                ),
            }
        }
        Err(err) => {
            let safe = err.replace('"', "'");
            HttpResponse::with_status_json(401, "Unauthorized", format!("{{\"error\":\"{safe}\"}}"))
        }
    }
}

/// Handle `POST /v1/auth/github`: exchange a GitHub OAuth code for a session.
fn handle_auth_github(request: &HttpRequest, auth: &Auth) -> HttpResponse {
    let code = match parse_json_field(&request.body, "code") {
        Some(c) => c,
        None => {
            return HttpResponse::with_status_json(
                400,
                "Bad Request",
                "{\"error\":\"expected JSON body with code\"}",
            )
        }
    };

    // `redirectUri` is optional; when present it must match the value the
    // browser used to start the flow (GitHub validates it).
    let redirect_uri = parse_json_field(&request.body, "redirectUri");

    match auth.exchange_github_code(&code, redirect_uri.as_deref()) {
        Ok(info) => {
            let exchange = auth.create_session(info);
            match serde_json::to_vec(&exchange) {
                Ok(json) => HttpResponse::json(json),
                Err(err) => HttpResponse::with_status_json(
                    500,
                    "Internal Server Error",
                    format!("{{\"error\":\"serialize: {err}\"}}"),
                ),
            }
        }
        Err(err) => {
            let safe = err.replace('"', "'");
            HttpResponse::with_status_json(401, "Unauthorized", format!("{{\"error\":\"{safe}\"}}"))
        }
    }
}

/// Handle `POST /v1/auth/refresh`: exchange a refresh token for fresh tokens.
fn handle_auth_refresh(request: &HttpRequest, auth: &Auth) -> HttpResponse {
    let refresh_token = match parse_json_field(&request.body, "refreshToken") {
        Some(t) => t,
        None => {
            return HttpResponse::with_status_json(
                400,
                "Bad Request",
                "{\"error\":\"expected JSON body with refreshToken\"}",
            )
        }
    };

    match auth.refresh(&refresh_token) {
        Some(exchange) => match serde_json::to_vec(&exchange) {
            Ok(json) => HttpResponse::json(json),
            Err(err) => HttpResponse::with_status_json(
                500,
                "Internal Server Error",
                format!("{{\"error\":\"serialize: {err}\"}}"),
            ),
        },
        None => HttpResponse::with_status_json(
            401,
            "Unauthorized",
            "{\"error\":\"invalid or expired refresh token\"}",
        ),
    }
}

/// True for a `/v1/conversations/{id}/messages` path.
fn is_messages_path(path: &str) -> bool {
    path.starts_with("/v1/conversations/") && path.ends_with("/messages")
}

/// Parse a named string field from a JSON body.
fn parse_json_field(body: &[u8], field: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get(field)
        .and_then(|t| t.as_str())
        .map(str::to_string)
}

/// Parse a `{"text":"..."}` request body, returning the message text.
fn parse_message_body(body: &[u8]) -> Option<String> {
    parse_json_field(body, "text")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Auth;

    /// Helper: build a dev-mode auth for tests.
    fn dev_auth() -> Auth {
        Auth::new(None)
    }

    #[test]
    fn recognizes_messages_path() {
        assert!(is_messages_path("/v1/conversations/abc/messages"));
        assert!(is_messages_path("/v1/conversations/conv-123/messages"));
        assert!(!is_messages_path("/v1/conversations/abc/stream"));
        assert!(!is_messages_path("/v1/ws/abc"));
    }

    #[test]
    fn parses_message_body() {
        assert_eq!(
            parse_message_body(br#"{"text":"hi"}"#),
            Some("hi".to_string())
        );
        assert_eq!(parse_message_body(br#"{"nope":1}"#), None);
        assert_eq!(parse_message_body(b"not json"), None);
    }

    #[test]
    fn parses_json_field() {
        assert_eq!(
            parse_json_field(br#"{"idToken":"abc"}"#, "idToken"),
            Some("abc".to_string())
        );
        assert_eq!(parse_json_field(br#"{"other":1}"#, "idToken"), None);
    }

    #[test]
    fn health_route_returns_ok() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/healthz".to_string(),
            ..Default::default()
        };
        let response = route(&request, &engine, &auth, None);
        assert_eq!(response.status(), 200);
    }

    #[test]
    fn readyz_route_reports_ready_when_no_dependencies_configured() {
        // A skeleton engine wires no external clients, so every dependency is
        // reported as skipped and the service is considered ready (HTTP 200).
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/readyz".to_string(),
            ..Default::default()
        };
        let response = route(&request, &engine, &auth, None);
        assert_eq!(response.status(), 200);
    }

    #[test]
    fn unknown_route_is_404() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/nope".to_string(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine, &auth, None).status(), 404);
    }

    #[test]
    fn non_api_get_is_served_by_the_static_site_when_present() {
        // With a static site configured, a non-API GET that has no matching file
        // returns the SPA shell (200) instead of a 404. API paths still 404.
        let mut root = std::env::temp_dir();
        root.push(format!("gaia-server-static-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("index.html"), b"<!doctype html>").unwrap();
        let site = StaticSite::new(root.clone());

        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let app_request = HttpRequest {
            method: "GET".to_string(),
            path: "/some/client/route".to_string(),
            ..Default::default()
        };
        assert_eq!(
            route(&app_request, &engine, &auth, Some(&site)).status(),
            200
        );

        // An unknown API path must NOT be masked by the SPA fallback.
        let api_request = HttpRequest {
            method: "GET".to_string(),
            path: "/v1/unknown".to_string(),
            ..Default::default()
        };
        assert_eq!(
            route(&api_request, &engine, &auth, Some(&site)).status(),
            404
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn message_route_runs_a_turn_for_authenticated_session() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        // Mint a session and send its access token: protected routes now require
        // a valid Google/GitHub session (no dev/guest fallback).
        let exchange = auth.create_session(crate::auth::UserInfo {
            sub: "user-1".to_string(),
            name: None,
            email: None,
            picture: None,
            github_login: None,
        });
        let mut headers = std::collections::BTreeMap::new();
        headers.insert(
            "authorization".to_string(),
            format!("Bearer {}", exchange.token),
        );
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/conversations/c1/messages".to_string(),
            body: br#"{"text":"hello"}"#.to_vec(),
            headers,
            ..Default::default()
        };
        let response = route(&request, &engine, &auth, None);
        assert_eq!(response.status(), 200);
    }

    #[test]
    fn message_route_without_auth_is_401() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/conversations/c1/messages".to_string(),
            body: br#"{"text":"hello"}"#.to_vec(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine, &auth, None).status(), 401);
    }

    #[test]
    fn auth_refresh_without_body_is_400() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/auth/refresh".to_string(),
            body: b"{}".to_vec(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine, &auth, None).status(), 400);
    }

    #[test]
    fn auth_google_without_body_is_400() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/auth/google".to_string(),
            body: b"{}".to_vec(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine, &auth, None).status(), 400);
    }

    #[test]
    fn auth_github_without_body_is_400() {
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/auth/github".to_string(),
            body: b"{}".to_vec(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine, &auth, None).status(), 400);
    }

    #[test]
    fn auth_refresh_with_a_valid_token_returns_fresh_tokens() {
        // Mint a session, then exchange its refresh token through the route: the
        // happy path serializes a fresh token pair with a 200.
        let engine = Engine::new(None, None);
        let auth = dev_auth();
        let exchange = auth.create_session(crate::auth::UserInfo {
            sub: "user-1".to_string(),
            name: None,
            email: None,
            picture: None,
            github_login: None,
        });
        let body = format!(r#"{{"refreshToken":"{}"}}"#, exchange.refresh_token);
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/auth/refresh".to_string(),
            body: body.into_bytes(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine, &auth, None).status(), 200);
    }

    #[test]
    fn cors_adds_permissive_headers() {
        let mut rendered = Vec::new();
        cors(HttpResponse::text(200, "OK", "ok"))
            .write_to(&mut rendered)
            .expect("render response");
        let rendered = String::from_utf8(rendered).expect("utf8 response");
        assert!(rendered.contains("Access-Control-Allow-Origin: *"));
        assert!(rendered.contains("Access-Control-Allow-Methods: GET, POST, OPTIONS"));
        assert!(rendered.contains("Access-Control-Allow-Headers: Authorization, Content-Type"));
    }

    #[test]
    fn to_json_serializes_a_frame() {
        let frame = TokenFrame {
            kind: "token",
            token: "hello",
        };
        assert_eq!(to_json(&frame), r#"{"type":"token","token":"hello"}"#);
    }

    #[test]
    fn stream_turn_writes_a_token_frame_then_a_done_frame() {
        use std::io::Read;
        use std::net::{TcpListener, TcpStream};

        // A real localhost socket pair: stream_turn writes WebSocket frames into
        // `client`, and we read them back off the accepted `server` end.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut client = TcpStream::connect(addr).expect("connect");
        let (mut server, _) = listener.accept().expect("accept");

        // A skeleton turn is a cheap, fully-formed TurnResult to stream.
        let result = Engine::new(None, None).run_turn("alice", "hi");
        stream_turn(&mut client, &result).expect("stream_turn writes both frames");
        client
            .shutdown(std::net::Shutdown::Write)
            .expect("close write half");

        let mut received = Vec::new();
        server.read_to_end(&mut received).expect("read frames");
        // Server text frames are unmasked, so the JSON payloads appear verbatim.
        let text = String::from_utf8_lossy(&received);
        assert!(text.contains(r#""type":"token""#), "missing token frame");
        assert!(text.contains(r#""type":"done""#), "missing done frame");
    }
}
