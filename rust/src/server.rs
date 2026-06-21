//! The [`Server`] type: Gaia's HTTP + WebSocket backend.
//!
//! The server exposes exactly the endpoints the PWA front end calls (after the
//! deprecated SSE streaming endpoint was removed):
//!
//! | Method | Path | Purpose |
//! |---|---|---|
//! | `GET`  | `/healthz`, `/readyz` | liveness / readiness |
//! | `POST` | `/v1/conversations/{id}/messages` | run one turn, return the reply |
//! | `GET`  | `/v1/ws/{id}` (Upgrade) | run one turn, stream the reply over WS |
//! | `POST` | `/v1/auth/google` | exchange a Google ID token for a session |
//! | `POST` | `/v1/auth/github` | exchange a GitHub OAuth code for a session |
//! | `POST` | `/v1/auth/refresh` | exchange a refresh token for a fresh session |
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
use crate::websocket::{self, Message};

/// How long a single connection may stay idle before we give up on it. Keeps
/// dropped/abandoned sockets from tying up a thread forever.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(120);

/// Gaia's backend HTTP/WebSocket server.
///
/// Holds a shared [`Engine`] (the turn runner) and [`Auth`] (session manager)
/// behind [`Arc`]s so every connection thread can run turns concurrently.
#[derive(Debug, Clone)]
pub struct Server {
    engine: Arc<Engine>,
    auth: Arc<Auth>,
}

impl Server {
    /// Build a server around an already-configured engine and auth manager.
    pub fn new(engine: Engine, auth: Auth) -> Self {
        Server {
            engine: Arc::new(engine),
            auth: Arc::new(auth),
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
                    // One thread per connection keeps the model blocking-simple.
                    std::thread::spawn(move || {
                        if let Err(err) = handle_connection(stream, &engine, &auth) {
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
fn handle_connection(stream: TcpStream, engine: &Engine, auth: &Auth) -> std::io::Result<()> {
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

    let response = cors(route(&request, engine, auth));
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
fn route(request: &HttpRequest, engine: &Engine, auth: &Auth) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        // Health / readiness probes (no auth).
        ("GET", "/healthz") | ("GET", "/readyz") => HttpResponse::text(200, "OK", "ok"),

        // Run one turn and return the reply.
        ("POST", path) if is_messages_path(path) => handle_message(request, engine, auth),

        // Exchange a Google ID token for a Gaia session.
        ("POST", "/v1/auth/google") => handle_auth_google(request, auth),

        // Exchange a GitHub OAuth code for a Gaia session.
        ("POST", "/v1/auth/github") => handle_auth_github(request, auth),

        // Refresh an expired access token.
        ("POST", "/v1/auth/refresh") => handle_auth_refresh(request, auth),

        // Unknown route.
        _ => HttpResponse::with_status_json(404, "Not Found", r#"{"error":"not found"}"#),
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
        let response = route(&request, &engine, &auth);
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
        assert_eq!(route(&request, &engine, &auth).status(), 404);
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
        let response = route(&request, &engine, &auth);
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
        assert_eq!(route(&request, &engine, &auth).status(), 401);
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
        assert_eq!(route(&request, &engine, &auth).status(), 400);
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
        assert_eq!(route(&request, &engine, &auth).status(), 400);
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
        assert_eq!(route(&request, &engine, &auth).status(), 400);
    }
}
