//! The [`HttpRequest`] type: a parsed HTTP/1.1 request.
//!
//! Gaia's backend speaks just enough HTTP/1.1 to serve a small JSON API and a
//! WebSocket upgrade, so — in the same spirit as the hand-rolled Cosmos client
//! ([`crate::cosmos`]) — we parse requests ourselves over a plain
//! [`std::net::TcpStream`] instead of taking a web-framework dependency.
//!
//! The parser is intentionally minimal: a request line, headers, and an
//! optional `Content-Length`-delimited body. That covers every request the
//! front end makes (`POST` JSON, `GET` health checks, and the WebSocket
//! `GET ... Upgrade`). Chunked request bodies and HTTP/2 are out of scope.

use std::collections::BTreeMap;
use std::io::{self, BufRead};

/// A parsed HTTP request: method, target, headers, query, and body.
///
/// Header names are stored lowercased so lookups are case-insensitive (HTTP
/// header names are case-insensitive by spec). The `path` and `query` values are
/// percent-decoded; `target` keeps the original raw request target for logging.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HttpRequest {
    /// The request method, uppercased (e.g. `GET`, `POST`, `OPTIONS`).
    pub method: String,
    /// The raw request target exactly as sent, e.g. `/v1/x?text=hi`.
    pub target: String,
    /// The percent-decoded path portion of the target, e.g. `/v1/x`.
    pub path: String,
    /// The percent-decoded query parameters, keyed by name.
    pub query: BTreeMap<String, String>,
    /// Request headers, with lowercased names for case-insensitive lookup.
    pub headers: BTreeMap<String, String>,
    /// The request body bytes (empty when there is no body).
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Read and parse a single request from `reader`.
    ///
    /// Returns `Ok(None)` on a clean end-of-stream before any request line was
    /// received (the peer closed an idle keep-alive connection), and
    /// `Ok(Some(request))` once a full request has been read. Malformed input or
    /// an I/O failure surfaces as `Err`.
    pub fn read<R: BufRead>(reader: &mut R) -> io::Result<Option<HttpRequest>> {
        // --- Request line ---------------------------------------------------
        let mut request_line = String::new();
        let read = reader.read_line(&mut request_line)?;
        if read == 0 {
            // Peer closed the connection without sending anything.
            return Ok(None);
        }

        let request_line = request_line.trim_end();
        let mut parts = request_line.split_whitespace();
        let method = parts
            .next()
            .ok_or_else(|| invalid("missing request method"))?
            .to_ascii_uppercase();
        let target = parts
            .next()
            .ok_or_else(|| invalid("missing request target"))?
            .to_string();

        // --- Headers --------------------------------------------------------
        let mut headers = BTreeMap::new();
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                // EOF mid-headers: treat as a truncated/invalid request.
                return Err(invalid("unexpected end of headers"));
            }
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break; // Blank line terminates the header block.
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }

        // --- Body (Content-Length only) ------------------------------------
        let mut body = Vec::new();
        if let Some(len) = headers
            .get("content-length")
            .and_then(|v| v.parse::<usize>().ok())
        {
            body.resize(len, 0);
            reader.read_exact(&mut body)?;
        }

        // --- Split + decode the target -------------------------------------
        let (raw_path, raw_query) = match target.split_once('?') {
            Some((p, q)) => (p, q),
            None => (target.as_str(), ""),
        };
        let path = percent_decode(raw_path);
        let query = parse_query(raw_query);

        Ok(Some(HttpRequest {
            method,
            target,
            path,
            query,
            headers,
            body,
        }))
    }

    /// Look up a header value by (case-insensitive) name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// The bearer token from the `Authorization: Bearer <token>` header, if any.
    pub fn bearer_token(&self) -> Option<&str> {
        let value = self.header("authorization")?;
        value
            .strip_prefix("Bearer ")
            .or_else(|| value.strip_prefix("bearer "))
    }

    /// True when this request is a WebSocket upgrade handshake.
    ///
    /// We require the `Upgrade: websocket` token and the presence of a
    /// `Sec-WebSocket-Key`, which together identify an RFC 6455 opening
    /// handshake regardless of header casing.
    pub fn is_websocket_upgrade(&self) -> bool {
        let upgrade = self
            .header("upgrade")
            .map(|v| v.to_ascii_lowercase().contains("websocket"))
            .unwrap_or(false);
        upgrade && self.header("sec-websocket-key").is_some()
    }
}

/// Construct an `io::Error` for a malformed request.
fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Parse an `a=1&b=2` query string into decoded key/value pairs.
fn parse_query(raw: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for pair in raw.split('&').filter(|s| !s.is_empty()) {
        let (key, value) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        map.insert(percent_decode(key), percent_decode(value));
    }
    map
}

/// Percent-decode a URL component, also turning `+` into a space.
///
/// Invalid `%` escapes are passed through literally rather than erroring, which
/// keeps the parser forgiving of odd input without panicking.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_val(bytes[i + 1]);
                let lo = hex_val(bytes[i + 2]);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    // Not a valid escape: keep the '%' literally and move on.
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    // The decoded bytes should be UTF-8 for our inputs; fall back lossily.
    String::from_utf8_lossy(&out).into_owned()
}

/// Map an ASCII hex digit to its 0-15 value, or `None` if it is not hex.
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Read a full request from raw bytes, used by tests and by callers that already
/// hold the bytes. Thin wrapper over [`HttpRequest::read`].
#[cfg(test)]
pub fn parse_bytes(bytes: &[u8]) -> io::Result<Option<HttpRequest>> {
    let mut cursor = io::BufReader::new(bytes);
    HttpRequest::read(&mut cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_post_with_json_body() {
        let raw = b"POST /v1/conversations/abc/messages HTTP/1.1\r\n\
            Host: localhost\r\n\
            Content-Type: application/json\r\n\
            Authorization: Bearer dev:alice\r\n\
            Content-Length: 14\r\n\
            \r\n\
            {\"text\":\"hi\"}\n";
        let req = parse_bytes(raw).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/conversations/abc/messages");
        assert_eq!(req.bearer_token(), Some("dev:alice"));
        assert_eq!(req.body.len(), 14);
        assert!(req.body.starts_with(b"{\"text\""));
    }

    #[test]
    fn decodes_query_parameters() {
        let raw = b"GET /v1/x?text=hello%20world&n=3 HTTP/1.1\r\nHost: x\r\n\r\n";
        let req = parse_bytes(raw).unwrap().unwrap();
        assert_eq!(req.path, "/v1/x");
        assert_eq!(
            req.query.get("text").map(String::as_str),
            Some("hello world")
        );
        assert_eq!(req.query.get("n").map(String::as_str), Some("3"));
    }

    #[test]
    fn detects_websocket_upgrade() {
        let raw = b"GET /v1/ws/abc HTTP/1.1\r\n\
            Host: localhost\r\n\
            Upgrade: websocket\r\n\
            Connection: Upgrade\r\n\
            Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
            Sec-WebSocket-Version: 13\r\n\
            \r\n";
        let req = parse_bytes(raw).unwrap().unwrap();
        assert!(req.is_websocket_upgrade());
        assert_eq!(
            req.header("sec-websocket-key"),
            Some("dGhlIHNhbXBsZSBub25jZQ==")
        );
    }

    #[test]
    fn clean_eof_returns_none() {
        assert!(parse_bytes(b"").unwrap().is_none());
    }
}
