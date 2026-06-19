//! Minimal **SHA-1** implementation (FIPS 180-4), standard library only.
//!
//! Like [`crate::base64`], this exists for exactly one reason: the WebSocket
//! opening handshake. RFC 6455 requires the server to answer a client's
//! `Sec-WebSocket-Key` with `base64( sha1( key + magic-GUID ) )`. That single
//! use does not justify adding a hashing crate to the dependency tree, so we
//! hand-roll the algorithm here (consistent with the project's preference for
//! the standard library and the hand-rolled protocol code in
//! [`crate::cosmos`]).
//!
//! SHA-1 is **broken for cryptographic purposes** and must never be used for
//! security here. It is used only because the WebSocket spec mandates it as a
//! fixed, non-secret handshake transform.

/// Compute the SHA-1 digest of `input`, returning the raw 20-byte hash.
///
/// This is a straight transcription of the FIPS 180-4 reference algorithm:
/// pad the message, process it in 512-bit (64-byte) blocks, and run the 80-round
/// compression function over five 32-bit working variables.
pub fn digest(input: &[u8]) -> [u8; 20] {
    // Initial hash values (FIPS 180-4, section 5.3.1).
    let mut h: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];

    // --- Pre-processing: pad the message ----------------------------------
    // Append 0x80, then zeros, then the 64-bit big-endian bit length, so the
    // total length is a multiple of 64 bytes.
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0x00);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    // --- Process each 64-byte block ---------------------------------------
    for block in message.chunks_exact(64) {
        // Break the block into sixteen big-endian 32-bit words, then extend to
        // the full 80-word message schedule.
        let mut w = [0u32; 80];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        // Initialise the five working variables from the current hash state.
        let [mut a, mut b, mut c, mut d, mut e] = h;

        // The 80 compression rounds, in four groups of twenty with different
        // mixing functions `f` and round constants `k`.
        for (i, &word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        // Fold the working variables back into the running hash state.
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }

    // Serialise the five state words big-endian into the 20-byte digest.
    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a digest as lowercase hex for comparison against published vectors.
    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    #[test]
    fn empty_input_matches_known_vector() {
        assert_eq!(
            hex(&digest(b"")),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
    }

    #[test]
    fn abc_matches_fips_vector() {
        // The classic FIPS 180-4 example.
        assert_eq!(
            hex(&digest(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
    }

    #[test]
    fn multi_block_input_matches_known_vector() {
        // 56 bytes forces a second padded block, exercising the block loop.
        let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(&digest(input)),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }
}
