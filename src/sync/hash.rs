//! Content-addressed hashing for change detection.
//!
//! Uses BLAKE3 for fast, cryptographically secure hashing with SIMD acceleration.
//! Content hashes are used to detect changes efficiently during sync.

use std::fmt;
use std::io::{self, Read};
use thiserror::Error;

/// A 256-bit content hash representing the identity of a piece of content.
/// 
/// Uses BLAKE3 which is:
/// - Faster than SHA-256, SHA-3, and BLAKE2
/// - SIMD-accelerated on x86_64 and ARM
/// - Cryptographically secure
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// The length of the hash in bytes.
    pub const LEN: usize = 32;

    /// Create a ContentHash from raw bytes.
    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes of the hash.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to a hex string (lowercase).
    pub fn to_hex(&self) -> String {
        // Pre-allocate exact size needed
        let mut hex = String::with_capacity(64);
        for byte in &self.0 {
            use std::fmt::Write;
            write!(hex, "{:02x}", byte).unwrap();
        }
        hex
    }

    /// Parse from a hex string.
    pub fn from_hex(s: &str) -> Result<Self, HashError> {
        if s.len() != 64 {
            return Err(HashError::InvalidLength {
                expected: 64,
                actual: s.len(),
            });
        }

        let mut bytes = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks_exact(2).enumerate() {
            // SAFETY: chunks_exact guarantees 2 bytes
            let high = hex_digit(chunk[0]).ok_or(HashError::InvalidHexChar { pos: i * 2 })?;
            let low = hex_digit(chunk[1]).ok_or(HashError::InvalidHexChar { pos: i * 2 + 1 })?;
            bytes[i] = (high << 4) | low;
        }

        Ok(Self(bytes))
    }

    /// Check if this hash matches another (constant-time comparison).
    #[inline]
    pub fn eq_ct(&self, other: &Self) -> bool {
        // Use constant_time_eq from blake3 internals
        constant_time_eq::constant_time_eq_n(&self.0, &other.0)
    }
}

/// Parse a single hex digit.
#[inline]
const fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({}…)", &self.to_hex()[..16])
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl serde::Serialize for ContentHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for ContentHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Errors that can occur when working with content hashes.
#[derive(Debug, Clone, Error)]
pub enum HashError {
    /// Hex string has wrong length.
    #[error("invalid hash length: expected {expected}, got {actual}")]
    InvalidLength { expected: usize, actual: usize },

    /// Hex string contains invalid character at position.
    #[error("invalid hex character at position {pos}")]
    InvalidHexChar { pos: usize },
}

/// Hash content bytes using BLAKE3.
/// 
/// This is the fastest option for in-memory data.
#[inline]
pub fn hash_content(content: &[u8]) -> ContentHash {
    ContentHash(*blake3::hash(content).as_bytes())
}

/// Hash content from a reader using BLAKE3.
/// 
/// Uses incremental hashing for memory efficiency with large files.
/// Reads in 64KB chunks (optimal for BLAKE3's internal buffer).
pub fn hash_reader<R: Read>(mut reader: R) -> io::Result<ContentHash> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 65536]; // 64KB - matches BLAKE3's internal buffer

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(ContentHash(*hasher.finalize().as_bytes()))
}

/// Hash multiple pieces of content together.
/// 
/// More efficient than concatenating and hashing.
#[inline]
pub fn hash_multi<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    ContentHash(*hasher.finalize().as_bytes())
}

/// Create a keyed hash (MAC) for authenticated content.
/// 
/// Useful for verifying content from untrusted sources.
#[inline]
pub fn hash_keyed(key: &[u8; 32], content: &[u8]) -> ContentHash {
    ContentHash(*blake3::keyed_hash(key, content).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let content = b"Hello, world!";
        let hash1 = hash_content(content);
        let hash2 = hash_content(content);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_different_content() {
        let hash1 = hash_content(b"Hello");
        let hash2 = hash_content(b"World");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hex_roundtrip() {
        let hash = hash_content(b"test");
        let hex = hash.to_hex();
        assert_eq!(hex.len(), 64);
        let parsed = ContentHash::from_hex(&hex).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_invalid_hex_length() {
        let result = ContentHash::from_hex("abc");
        assert!(matches!(
            result,
            Err(HashError::InvalidLength {
                expected: 64,
                actual: 3
            })
        ));
    }

    #[test]
    fn test_invalid_hex_chars() {
        let result = ContentHash::from_hex(&"g".repeat(64));
        assert!(matches!(result, Err(HashError::InvalidHexChar { pos: 0 })));
    }

    #[test]
    fn test_hash_reader() {
        let data = b"Hello, world!";
        let hash1 = hash_content(data);
        let hash2 = hash_reader(&data[..]).unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_multi() {
        let hash1 = hash_content(b"HelloWorld");
        let hash2 = hash_multi([b"Hello".as_slice(), b"World".as_slice()]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_keyed() {
        let key = [0u8; 32];
        let hash1 = hash_keyed(&key, b"test");
        let hash2 = hash_keyed(&key, b"test");
        assert_eq!(hash1, hash2);

        // Different key = different hash
        let key2 = [1u8; 32];
        let hash3 = hash_keyed(&key2, b"test");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_constant_time_eq() {
        let hash1 = hash_content(b"test");
        let hash2 = hash_content(b"test");
        let hash3 = hash_content(b"other");

        assert!(hash1.eq_ct(&hash2));
        assert!(!hash1.eq_ct(&hash3));
    }

    #[test]
    fn test_serde_roundtrip() {
        let hash = hash_content(b"test");
        let json = serde_json::to_string(&hash).unwrap();
        let parsed: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_known_vector() {
        // BLAKE3 test vector from spec
        let hash = hash_content(b"");
        assert_eq!(
            hash.to_hex(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }
}
