//! Minimal standard Base64 **encoder** (RFC 4648), standard library only.
//!
//! Gaia's HTTP server needs Base64 in exactly one place: the WebSocket opening
//! handshake, where the server must return
//! `Sec-WebSocket-Accept = base64( sha1( key + GUID ) )` (see
//! [`crate::websocket`]). Rather than pull in a crate for those few bytes, we
//! hand-roll the tiny encoder here, mirroring how [`crate::cosmos`] hand-rolls
//! the bits of protocol it needs instead of taking an SDK dependency.
//!
//! Only **encoding** is implemented because that is all the handshake requires;
//! there is deliberately no decoder.

/// The standard Base64 alphabet (RFC 4648, "standard" table, not URL-safe).
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes as a standard Base64 string, with `=` padding.
///
/// This is the classic 3-bytes-in / 4-chars-out transform: every group of three
/// input bytes becomes four output characters, and the final partial group is
/// padded with `=` so the output length is always a multiple of four.
///
/// # Examples
///
/// ```ignore
/// // (internal module; shown for illustration)
/// assert_eq!(base64::encode(b""), "");
/// assert_eq!(base64::encode(b"f"), "Zg==");
/// assert_eq!(base64::encode(b"foobar"), "Zm9vYmFy");
/// ```
pub fn encode(input: &[u8]) -> String {
    // Each 3-byte chunk yields 4 characters; pre-size the output to avoid
    // reallocations as we push.
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);

    for chunk in input.chunks(3) {
        // Pack up to three bytes into a 24-bit big-endian buffer. Missing bytes
        // (in the final short chunk) are treated as zero.
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        // Slice the 24-bit buffer into four 6-bit indices, most significant
        // first, and map each through the alphabet.
        out.push(ALPHABET[((triple >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3f) as usize] as char);

        // The last one or two characters become `=` padding when the input
        // chunk was short (1 or 2 bytes instead of 3).
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4648_test_vectors() {
        // The canonical examples from RFC 4648 section 10.
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"f"), "Zg==");
        assert_eq!(encode(b"fo"), "Zm8=");
        assert_eq!(encode(b"foo"), "Zm9v");
        assert_eq!(encode(b"foob"), "Zm9vYg==");
        assert_eq!(encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn encodes_arbitrary_bytes() {
        // A non-text byte sequence still round-trips to the expected length and
        // padding (4 chars per 3 input bytes).
        let bytes = [0x00u8, 0xff, 0x10, 0x20, 0x30];
        let encoded = encode(&bytes);
        assert_eq!(encoded.len(), 8); // 5 bytes -> 2 chunks -> 8 chars
        assert!(encoded.ends_with('='));
    }
}
