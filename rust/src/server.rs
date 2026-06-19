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
//! | `POST` | `/v1/auth/google`, `/v1/auth/refresh` | not implemented (dev uses bearer subjects) |
//!
//! It is a small, blocking, thread-per-connection server built directly on
//! [`std::net`] — no async runtime and no web framework — matching the project's
//! preference for the standard library and its hand-rolled protocol code
//! ([`crate::cosmos`]). Each connection is parsed with
//! [`crate::http_request::HttpRequest`], dispatched here, and answered with a
//! [`crate::http_response::HttpResponse`] (or upgraded to a WebSocket).
//!
//! **Auth model:** in dev-auth mode the front end sends
//! `Authorization: Bearer dev:<name>` (or a `{token}` WebSocket hello), and we
//! map that subject to the `user_id` every read/write is scoped to. Real Google
//! JWT verification is intentionally out of scope here.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use crate::engine::Engine;
use crate::http_request::HttpRequest;
use crate::http_response::HttpResponse;
use crate::websocket::{self, Message};

/// How long a single connection may stay idle before we give up on it. Keeps
/// dropped/abandoned sockets from tying up a thread forever.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(120);

/// Gaia's backend HTTP/WebSocket server.
///
/// Holds a shared [`Engine`] (the turn runner) behind an [`Arc`] so every
/// connection thread can run turns concurrently without cloning the model
/// configuration per request.
#[derive(Debug, Clone)]
pub struct Server {
    engine: Arc<Engine>,
}

impl Server {
    /// Build a server around an already-configured engine.
    pub fn new(engine: Engine) -> Self {
        Server {
            engine: Arc::new(engine),
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
                    // One thread per connection keeps the model blocking-simple.
                    std::thread::spawn(move || {
                        if let Err(err) = handle_connection(stream, &engine) {
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
fn handle_connection(stream: TcpStream, engine: &Engine) -> std::io::Result<()> {
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
        return handle_websocket(&mut reader, &mut writer, &request, engine);
    }

    let response = cors(route(&request, engine));
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
fn route(request: &HttpRequest, engine: &Engine) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        // Health / readiness probes (no auth).
        ("GET", "/healthz") | ("GET", "/readyz") => HttpResponse::text(200, "OK", "ok"),

        // Run one turn and return the reply.
        ("POST", path) if is_messages_path(path) => handle_message(request, engine),

        // Auth endpoints are only used by the Google sign-in flow, which this
        // backend does not implement; dev-auth uses bearer subjects directly.
        ("POST", "/v1/auth/google") | ("POST", "/v1/auth/refresh") => {
            HttpResponse::with_status_json(
                501,
                "Not Implemented",
                r#"{"error":"google auth is not implemented; use dev-auth (Bearer dev:<name>)"}"#,
            )
        }

        // Unknown route.
        _ => HttpResponse::with_status_json(404, "Not Found", r#"{"error":"not found"}"#),
    }
}

/// Handle `POST /v1/conversations/{id}/messages`: parse the body, run the turn.
fn handle_message(request: &HttpRequest, engine: &Engine) -> HttpResponse {
    let text = match parse_message_body(&request.body) {
        Some(text) => text,
        None => {
            return HttpResponse::with_status_json(
                400,
                "Bad Request",
                r#"{"error":"expected JSON body {\"text\":\"...\"}"}"#,
            )
        }
    };

    let user_id = user_id_from_request(request);
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

    // Default the user to the request's bearer (if any) or the dev id; the
    // client's `{token}` hello refines it before the turn runs.
    let mut user_id = user_id_from_request(request);

    while let Some(message) = websocket::read_message(reader)? {
        match message {
            Message::Text(text) => {
                let value: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(_) => continue, // ignore non-JSON frames
                };

                // A hello frame carries the auth token; map it to the user id.
                if let Some(token) = value.get("token").and_then(|t| t.as_str()) {
                    user_id = subject_from_token(token);
                }

                // A text frame carries the user's message; run the turn.
                if let Some(input) = value.get("text").and_then(|t| t.as_str()) {
                    let result = engine.run_turn(&user_id, input);
                    stream_turn(writer, &result)?;
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

/// True for a `/v1/conversations/{id}/messages` path.
fn is_messages_path(path: &str) -> bool {
    path.starts_with("/v1/conversations/") && path.ends_with("/messages")
}

/// Extract the user id a request's turn should be scoped to.
fn user_id_from_request(request: &HttpRequest) -> String {
    match request.bearer_token() {
        Some(token) => subject_from_token(token),
        // No bearer: fall back to the configured dev user so isolation still has
        // a concrete subject to scope reads/writes to.
        None => crate::llm::dev_user_id(),
    }
}

/// Reduce a bearer token to its subject (the `user_id`).
///
/// Dev-auth tokens look like `dev:<name>`; we use `<name>`. Anything else is
/// used verbatim as the subject (we do not verify or decode JWTs here).
fn subject_from_token(token: &str) -> String {
    let token = token.trim();
    token.strip_prefix("dev:").unwrap_or(token).to_string()
}

/// Parse a `{"text":"..."}` request body, returning the message text.
fn parse_message_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("text")
        .and_then(|t| t.as_str())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_messages_path() {
        assert!(is_messages_path("/v1/conversations/abc/messages"));
        assert!(is_messages_path("/v1/conversations/conv-123/messages"));
        assert!(!is_messages_path("/v1/conversations/abc/stream"));
        assert!(!is_messages_path("/v1/ws/abc"));
    }

    #[test]
    fn subject_strips_dev_prefix() {
        assert_eq!(subject_from_token("dev:alice"), "alice");
        assert_eq!(subject_from_token("  dev:bob  "), "bob");
        // A non-dev token is used as-is.
        assert_eq!(subject_from_token("opaque-jwt"), "opaque-jwt");
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
    fn health_route_returns_ok() {
        let engine = Engine::new(None, None);
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/healthz".to_string(),
            ..Default::default()
        };
        let response = route(&request, &engine);
        assert_eq!(response.status(), 200);
    }

    #[test]
    fn unknown_route_is_404() {
        let engine = Engine::new(None, None);
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/nope".to_string(),
            ..Default::default()
        };
        assert_eq!(route(&request, &engine).status(), 404);
    }

    #[test]
    fn message_route_runs_a_turn_in_skeleton_mode() {
        let engine = Engine::new(None, None);
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/conversations/c1/messages".to_string(),
            body: br#"{"text":"hello"}"#.to_vec(),
            ..Default::default()
        };
        let response = route(&request, &engine);
        assert_eq!(response.status(), 200);
    }
}
