//! Sync protocol layer for distributed meta repository synchronization.
//!
//! Implements a layered sync model:
//! - **Layer 0 (L0)**: Canonical data - commits, documents, metadata
//! - **Layer 1 (L1)**: Embeddings - content-addressed vectors for semantic search  
//! - **Layer 2 (L2)**: Indices - HNSW search structures
//!
//! # Design Principles
//!
//! - L0 is always shipped (source of truth)
//! - L1/L2 can be shipped or rebuilt depending on peer capability
//! - Change detection uses content hashes for efficiency
//! - Tolerance-based verification for floating-point data

mod hash;
mod layer;
mod capability;

pub use hash::{ContentHash, hash_content};
pub use layer::{Layer, LayerKind};
pub use capability::{Capability, PeerCapability};

/// Protocol version for compatibility checking.
pub const PROTOCOL_VERSION: &str = "1.0.0-alpha";

/// Minimum cosine similarity for embedding verification.
/// Validated: cross-platform variance is ~10⁻⁸, well below this threshold.
pub const EMBEDDING_TOLERANCE: f64 = 0.9999;

/// Default HNSW parameters (validated on 10k real embeddings).
pub mod hnsw_defaults {
    /// Graph connectivity parameter.
    pub const M: usize = 16;
    /// Construction-time expansion factor.
    pub const EF_CONSTRUCTION: usize = 200;
    /// Search-time expansion factor.
    pub const EF_SEARCH: usize = 50;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert!(PROTOCOL_VERSION.starts_with("1."));
    }

    #[test]
    fn test_embedding_tolerance() {
        assert!(EMBEDDING_TOLERANCE > 0.99);
        assert!(EMBEDDING_TOLERANCE < 1.0);
    }

    #[test]
    fn test_hnsw_defaults() {
        assert_eq!(hnsw_defaults::M, 16);
        assert_eq!(hnsw_defaults::EF_CONSTRUCTION, 200);
        assert_eq!(hnsw_defaults::EF_SEARCH, 50);
    }
}
