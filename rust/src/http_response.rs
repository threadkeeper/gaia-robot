//! The [`HttpResponse`] type: a small HTTP/1.1 response builder + writer.
//!
//! Pairs with [`crate::http_request::HttpRequest`] to give Gaia's backend just
//! enough of HTTP to answer JSON and health-check requests over a plain socket,
//! without a web-framework dependency. Responses always carry an explicit
//! `Content-Length` and close the connection (`Connection: close`) so the framing
//! stays trivial and unambiguous.

use std::io::{self, Write};

/// A buffered HTTP response: status, headers, and a body, ready to write.
///
/// Build one with a constructor like [`HttpResponse::json`], optionally attach
/// extra headers with [`HttpResponse::with_header`], then send it with
/// [`HttpResponse::write_to`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// Numeric status code, e.g. `200`.
    status: u16,
    /// Reason phrase, e.g. `OK`.
    reason: &'static str,
    /// Response headers as ordered `(name, value)` pairs.
    headers: Vec<(String, String)>,
    /// The response body bytes.
    body: Vec<u8>,
}

impl HttpResponse {
    /// A `200 OK` JSON response carrying `body` verbatim as the payload.
    pub fn json(body: impl Into<Vec<u8>>) -> Self {
        Self::with_status_json(200, "OK", body)
    }

    /// A JSON response with an explicit status code and reason phrase.
    pub fn with_status_json(status: u16, reason: &'static str, body: impl Into<Vec<u8>>) -> Self {
        HttpResponse {
            status,
            reason,
            headers: vec![(
                "Content-Type".to_string(),
                "application/json; charset=utf-8".to_string(),
            )],
            body: body.into(),
        }
    }

    /// A `text/plain` response with an explicit status code and reason phrase.
    pub fn text(status: u16, reason: &'static str, body: impl Into<Vec<u8>>) -> Self {
        HttpResponse {
            status,
            reason,
            headers: vec![(
                "Content-Type".to_string(),
                "text/plain; charset=utf-8".to_string(),
            )],
            body: body.into(),
        }
    }

    /// An empty response (no body) with the given status and reason.
    pub fn empty(status: u16, reason: &'static str) -> Self {
        HttpResponse {
            status,
            reason,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Attach an extra response header, returning `self` for chaining.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The status code (exposed for logging and tests).
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Serialize the full response (status line, headers, blank line, body) and
    /// write it to `w`.
    ///
    /// A `Content-Length` and `Connection: close` are always emitted; any
    /// `Content-Length` the caller added via [`HttpResponse::with_header`] is
    /// ignored in favour of the real body length to keep framing correct.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut head = format!("HTTP/1.1 {} {}\r\n", self.status, self.reason);

        for (name, value) in &self.headers {
            // Skip a caller-supplied Content-Length; we set the authoritative one.
            if name.eq_ignore_ascii_case("content-length") {
                continue;
            }
            head.push_str(&format!("{name}: {value}\r\n"));
        }

        head.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        head.push_str("Connection: close\r\n\r\n");

        w.write_all(head.as_bytes())?;
        w.write_all(&self.body)?;
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_status_line_headers_and_body() {
        let resp = HttpResponse::json("{\"ok\":true}").with_header("X-Test", "1");
        let mut buf = Vec::new();
        resp.write_to(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();

        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Type: application/json; charset=utf-8\r\n"));
        assert!(text.contains("X-Test: 1\r\n"));
        assert!(text.contains("Content-Length: 11\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"ok\":true}"));
    }

    #[test]
    fn caller_content_length_is_overridden() {
        // A bogus caller-supplied Content-Length must not reach the wire.
        let resp =
            HttpResponse::text(404, "Not Found", "nope").with_header("Content-Length", "999");
        let mut buf = Vec::new();
        resp.write_to(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("Content-Length: 4\r\n"));
        assert!(!text.contains("999"));
    }

    #[test]
    fn empty_response_has_no_body_and_carries_the_status() {
        let resp = HttpResponse::empty(204, "No Content");
        let mut buf = Vec::new();
        resp.write_to(&mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();

        assert!(text.starts_with("HTTP/1.1 204 No Content\r\n"));
        assert!(text.contains("Content-Length: 0\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }
}
