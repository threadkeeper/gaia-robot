//! Shared test-only helper: a tiny, std-only one-shot HTTP server.
//!
//! Several of Gaia's clients ([`crate::web_search::BraveClient`],
//! [`crate::llm::LlmClient`], [`crate::embeddings::EmbeddingClient`], and
//! [`crate::cosmos::CosmosClient`]) talk to an HTTP endpoint with [`ureq`]. To
//! exercise their real request/response paths in unit tests — without a network
//! dependency or a mocking crate — we point them at a localhost server that
//! accepts a single connection and replies with a fixed status and body.
//!
//! The whole module is compiled only for tests (it is declared
//! `#[cfg(test)] mod test_http;` in `main.rs`), so it never ships in a release
//! build and adds no production dependency.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread::{self, JoinHandle};

/// Spin up a one-shot local HTTP server that replies with `status_line` and
/// `body`, returning its `http://127.0.0.1:<port>/` base URL and the join
/// handle for the serving thread.
///
/// It accepts exactly one connection, fully drains the request (so a half-read
/// socket never triggers a connection reset on Windows), writes a single fixed
/// response, then closes cleanly. Call [`JoinHandle::join`] at the end of the
/// test to make sure the server thread finished.
pub(crate) fn spawn_mock_http(
    status_line: &'static str,
    body: &'static str,
) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            // Read the whole request before answering. Closing a socket that
            // still has unread bytes makes Windows send an RST, which the client
            // reports as a broken pipe while reading the status line. Draining
            // the request avoids that race.
            let mut data: Vec<u8> = Vec::new();
            let mut buf = [0u8; 1024];

            // First, read until we have the full header block (CRLF CRLF).
            let header_end = loop {
                match stream.read(&mut buf) {
                    Ok(0) => break data.len(),
                    Ok(n) => {
                        data.extend_from_slice(&buf[..n]);
                        if let Some(pos) = find_subsequence(&data, b"\r\n\r\n") {
                            break pos + 4;
                        }
                    }
                    Err(_) => break data.len(),
                }
            };

            // Then read any Content-Length body so nothing is left unread.
            let headers = String::from_utf8_lossy(&data[..header_end.min(data.len())]);
            let content_len = headers
                .lines()
                .find_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            let mut body_seen = data.len().saturating_sub(header_end);
            while body_seen < content_len {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => body_seen += n,
                    Err(_) => break,
                }
            }

            // Write the fixed response and close the write half cleanly.
            let response = format!(
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);

            // Drain until the client closes its side to dodge an RST on drop.
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
        }
    });
    (format!("http://{addr}/"), handle)
}

/// Spin up a local HTTP server that answers a fixed *sequence* of requests,
/// one queued `(status_line, body)` response per request, in order.
///
/// Several flows make more than one HTTP call against the same client — for
/// example [`crate::engine::Engine::run_turn`] issues two LLM completions, and
/// [`crate::write_data_controller::WriteDataController::upsert_daily`] does a
/// point read, an upsert, and an index upsert. Each of Gaia's clients sets
/// `Connection: close`, so every call opens a fresh TCP connection; this server
/// therefore accepts exactly `responses.len()` connections and replies to each
/// with the next queued response.
///
/// Returns the `http://127.0.0.1:<port>/` base URL and the serving thread's
/// join handle. The test must make exactly as many requests as there are
/// responses, then [`JoinHandle::join`] the handle.
pub(crate) fn spawn_mock_http_sequence(
    responses: Vec<(String, String)>,
) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = thread::spawn(move || {
        for (status_line, body) in responses {
            match listener.accept() {
                Ok((mut stream, _)) => serve_one(&mut stream, &status_line, &body),
                Err(_) => break,
            }
        }
    });
    (format!("http://{addr}/"), handle)
}

/// Fully drain one request from `stream`, then write a single fixed response.
///
/// Shared by the one-shot and sequence servers: draining the request before
/// answering avoids a Windows RST-on-close race, and `Connection: close` tells
/// the client this connection is single-use.
fn serve_one(stream: &mut std::net::TcpStream, status_line: &str, body: &str) {
    let mut data: Vec<u8> = Vec::new();
    let mut buf = [0u8; 1024];

    // Read until the full header block (CRLF CRLF) has arrived.
    let header_end = loop {
        match stream.read(&mut buf) {
            Ok(0) => break data.len(),
            Ok(n) => {
                data.extend_from_slice(&buf[..n]);
                if let Some(pos) = find_subsequence(&data, b"\r\n\r\n") {
                    break pos + 4;
                }
            }
            Err(_) => break data.len(),
        }
    };

    // Then read any Content-Length body so nothing is left unread.
    let headers = String::from_utf8_lossy(&data[..header_end.min(data.len())]);
    let content_len = headers
        .lines()
        .find_map(|line| {
            let lower = line.to_ascii_lowercase();
            lower
                .strip_prefix("content-length:")
                .map(|value| value.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);
    let mut body_seen = data.len().saturating_sub(header_end);
    while body_seen < content_len {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => body_seen += n,
            Err(_) => break,
        }
    }

    let response = format!(
        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Write);

    // Drain until the client closes its side to dodge an RST on drop.
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(_) => break,
        }
    }
}

/// Find the first index of `needle` within `haystack`, or `None`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
