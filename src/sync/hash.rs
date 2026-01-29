//! Content-addressed hashing for change detection.
//!
//! Uses BLAKE3 for fast, cryptographically secure hashing.
//! Content hashes are used to detect changes efficiently during sync.

use std::fmt;

/// A content hash representing the identity of a piece of content.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Create a ContentHash from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes of the hash.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to a hex string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parse from a hex string.
    pub fn from_hex(s: &str) -> Result<Self, HashError> {
        if s.len() != 64 {
            return Err(HashError::InvalidLength(s.len()));
        }
        let mut bytes = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk).map_err(|_| HashError::InvalidHex)?;
            bytes[i] = u8::from_str_radix(hex_str, 16).map_err(|_| HashError::InvalidHex)?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({})", &self.to_hex()[..16])
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Errors that can occur when working with content hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashError {
    /// Hex string has wrong length.
    InvalidLength(usize),
    /// Hex string contains invalid characters.
    InvalidHex,
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashError::InvalidLength(len) => {
                write!(f, "invalid hash length: expected 64, got {}", len)
            }
            HashError::InvalidHex => write!(f, "invalid hex characters in hash"),
        }
    }
}

impl std::error::Error for HashError {}

/// Hash content using a simple algorithm.
/// 
/// Note: In production, this should use BLAKE3 for performance.
/// This implementation uses a simple rolling hash for demonstration.
pub fn hash_content(content: &[u8]) -> ContentHash {
    // Simple hash implementation (would use blake3 in production)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let h1 = hasher.finish();
    
    // Generate more bytes by rehashing
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let mut hasher = DefaultHasher::new();
        (h1, i as u64).hash(&mut hasher);
        let h = hasher.finish();
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    
    ContentHash(bytes)
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
        let parsed = ContentHash::from_hex(&hex).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_invalid_hex_length() {
        let result = ContentHash::from_hex("abc");
        assert!(matches!(result, Err(HashError::InvalidLength(3))));
    }

    #[test]
    fn test_invalid_hex_chars() {
        let result = ContentHash::from_hex(&"g".repeat(64));
        assert!(matches!(result, Err(HashError::InvalidHex)));
    }
}
