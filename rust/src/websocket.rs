//! WebSocket (RFC 6455) opening handshake + minimal framing.
//!
//! This is the streaming transport for Gaia's chat: after the front end removed
//! the deprecated SSE endpoint, replies are delivered either over a single
//! `POST` (non-streaming) or token-by-token over this WebSocket. The module is
//! deliberately small — it implements only what the chat needs:
//!
//! - [`accept_key`] computes the `Sec-WebSocket-Accept` handshake response.
//! - [`Message`] plus [`read_message`] decode a single client frame (text,
//!   ping, or close), unmasking the payload as the spec requires.
//! - [`write_text`] and [`write_close`] encode unmasked server frames.
//!
//! Message fragmentation and extensions are not supported: the small JSON
//! control messages the client sends (`{token}`, `{text}`) always arrive as a
//! single unfragmented text frame, which is all we read.

use std::io::{self, Read, Write};

use crate::{base64, sha1};

/// The fixed GUID RFC 6455 appends to the client key before hashing.
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// A decoded WebSocket message (one frame).
///
/// Only the variants Gaia acts on are modelled; binary frames are accepted but
/// carried opaquely, and continuation frames are not produced because we do not
/// support fragmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// A UTF-8 text frame (opcode `0x1`) — the JSON control messages.
    Text(String),
    /// A binary frame (opcode `0x2`).
    Binary(Vec<u8>),
    /// A ping control frame (opcode `0x9`); the caller should answer with a pong.
    Ping(Vec<u8>),
    /// A pong control frame (opcode `0xA`).
    Pong(Vec<u8>),
    /// A close control frame (opcode `0x8`).
    Close,
}

/// Compute the `Sec-WebSocket-Accept` value for a client's `Sec-WebSocket-Key`.
///
/// Per RFC 6455 the server answers `base64( sha1( key + WS_GUID ) )`. This is a
/// fixed, non-secret transform that proves the server understood the upgrade;
/// SHA-1 is used here only because the spec mandates it, not for security.
pub fn accept_key(client_key: &str) -> String {
    let mut combined = String::with_capacity(client_key.len() + WS_GUID.len());
    combined.push_str(client_key.trim());
    combined.push_str(WS_GUID);
    base64::encode(&sha1::digest(combined.as_bytes()))
}

/// Read a single WebSocket frame from `reader`.
///
/// Returns `Ok(None)` on a clean end-of-stream (the peer went away). Control and
/// data frames alike are returned as a [`Message`]; the caller decides how to
/// react (e.g. reply to a [`Message::Ping`] or stop on [`Message::Close`]).
///
/// Client frames are required by the spec to be masked; we unmask when the mask
/// bit is set and otherwise read the payload as-is, which keeps the reader
/// robust against lenient clients without weakening anything we rely on.
pub fn read_message<R: Read>(reader: &mut R) -> io::Result<Option<Message>> {
    // --- First two header bytes ----------------------------------------
    let mut header = [0u8; 2];
    if !read_exact_or_eof(reader, &mut header)? {
        return Ok(None);
    }
    let opcode = header[0] & 0x0f;
    let masked = header[1] & 0x80 != 0;
    let len_marker = (header[1] & 0x7f) as usize;

    // --- Extended payload length --------------------------------------
    let payload_len = match len_marker {
        126 => {
            let mut ext = [0u8; 2];
            reader.read_exact(&mut ext)?;
            u16::from_be_bytes(ext) as usize
        }
        127 => {
            let mut ext = [0u8; 8];
            reader.read_exact(&mut ext)?;
            u64::from_be_bytes(ext) as usize
        }
        other => other,
    };

    // --- Masking key (client -> server frames) ------------------------
    let mut mask = [0u8; 4];
    if masked {
        reader.read_exact(&mut mask)?;
    }

    // --- Payload, unmasked in place -----------------------------------
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        reader.read_exact(&mut payload)?;
        if masked {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
        }
    }

    let message = match opcode {
        0x1 => Message::Text(String::from_utf8_lossy(&payload).into_owned()),
        0x2 => Message::Binary(payload),
        0x8 => Message::Close,
        0x9 => Message::Ping(payload),
        0xA => Message::Pong(payload),
        // Continuation (0x0) or unknown opcodes: surface as binary so the caller
        // can choose to ignore them. We never emit fragmented frames ourselves.
        _ => Message::Binary(payload),
    };
    Ok(Some(message))
}

/// Encode and write an unmasked text frame (a complete, unfragmented message).
pub fn write_text<W: Write>(writer: &mut W, text: &str) -> io::Result<()> {
    write_frame(writer, 0x1, text.as_bytes())
}

/// Encode and write a close control frame (no status payload).
pub fn write_close<W: Write>(writer: &mut W) -> io::Result<()> {
    write_frame(writer, 0x8, &[])
}

/// Encode and write a pong control frame echoing `payload`.
pub fn write_pong<W: Write>(writer: &mut W, payload: &[u8]) -> io::Result<()> {
    write_frame(writer, 0xA, payload)
}

/// Write a single FIN frame with the given opcode and payload (server frames are
/// always unmasked, per the spec).
fn write_frame<W: Write>(writer: &mut W, opcode: u8, payload: &[u8]) -> io::Result<()> {
    let mut frame = Vec::with_capacity(payload.len() + 10);

    // FIN bit set (0x80) | opcode. We never fragment, so FIN is always 1.
    frame.push(0x80 | opcode);

    // Payload length, in one of the three RFC 6455 forms. The server never masks.
    let len = payload.len();
    if len <= 125 {
        frame.push(len as u8);
    } else if len <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(len as u64).to_be_bytes());
    }

    frame.extend_from_slice(payload);
    writer.write_all(&frame)?;
    writer.flush()
}

/// Read exactly `buf.len()` bytes, returning `Ok(false)` if the stream ended
/// cleanly before any byte was read (so the caller can treat it as EOF rather
/// than an error).
fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false), // clean EOF
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "partial frame",
                ))
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn accept_key_matches_rfc6455_example() {
        // The worked example from RFC 6455 section 1.3.
        let accept = accept_key("dGhlIHNhbXBsZSBub25jZQ==");
        assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn reads_a_masked_client_text_frame() {
        // A masked "Hi" text frame: FIN+text, MASK+len2, 4-byte mask, masked payload.
        let mask = [0x37u8, 0xfa, 0x21, 0x3d];
        let plain = b"Hi";
        let masked: Vec<u8> = plain
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ mask[i % 4])
            .collect();
        let mut frame = vec![0x81, 0x80 | 0x02];
        frame.extend_from_slice(&mask);
        frame.extend_from_slice(&masked);

        let mut cursor = Cursor::new(frame);
        let msg = read_message(&mut cursor).unwrap().unwrap();
        assert_eq!(msg, Message::Text("Hi".to_string()));
    }

    #[test]
    fn write_then_read_round_trips_text() {
        // Server writes an unmasked frame; reading it back yields the same text.
        let mut buf = Vec::new();
        write_text(&mut buf, "hello gaia").unwrap();

        let mut cursor = Cursor::new(buf);
        let msg = read_message(&mut cursor).unwrap().unwrap();
        assert_eq!(msg, Message::Text("hello gaia".to_string()));
    }

    #[test]
    fn clean_eof_yields_none() {
        let mut cursor = Cursor::new(Vec::new());
        assert!(read_message(&mut cursor).unwrap().is_none());
    }
}
