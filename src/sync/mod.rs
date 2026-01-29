//! Sync protocol layer for distributed meta repository synchronization.
//!
//! Implements a layered sync model:
//! - **Layer 0 (L0)**: Canonical data - commits, documents, metadata
//! - **Layer 1 (L1)**: Embeddings - content-addressed vectors for semantic search  
//! - **Layer 2 (L2)**: Indices - HNSW search structures
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  Layer 2: VectorIndex                   │  ← Can ship or rebuild
//! │  HNSW index, search structures          │
//! ├─────────────────────────────────────────┤
//! │  Layer 1: Embeddings                    │  ← Can ship or regenerate
//! │  Pre-computed vectors per chunk         │
//! ├─────────────────────────────────────────┤
//! │  Layer 0: Canonical Data                │  ← Always shipped
//! │  Commits, document hunks, metadata      │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Design Principles
//!
//! - L0 is always shipped (source of truth)
//! - L1/L2 can be shipped or rebuilt depending on peer capability
//! - Change detection uses BLAKE3 content hashes for efficiency
//! - Tolerance-based verification for floating-point data
//!
//! # Example
//!
//! ```
//! use meta_core::sync::{
//!     hash_content, Layer, LayerKind, LayerSet,
//!     PeerCapability, CapabilityTier, negotiate,
//! };
//!
//! // Create layer set with canonical data
//! let mut layers = LayerSet::new();
//! let hash = hash_content(b"document content");
//! layers.set_layer(Layer::new(LayerKind::Canonical, hash, 1024));
//!
//! // Negotiate sync between peers
//! let source = PeerCapability::new("server", CapabilityTier::Full);
//! let target = PeerCapability::new("client", CapabilityTier::Lite);
//! let plan = negotiate(&source, &target).unwrap();
//!
//! // Plan tells us what to ship vs. generate
//! assert!(plan.ship_layers.contains(&LayerKind::Canonical));
//! ```

mod capability;
mod hash;
mod layer;

pub use capability::{
    negotiate, negotiate_permissive, Capability, CapabilitySet, CapabilityTier, 
    NegotiationError, PeerCapability, SyncPlan,
};
pub use hash::{hash_content, hash_keyed, hash_multi, hash_reader, ContentHash, HashError};
pub use layer::{Layer, LayerDiff, LayerKind, LayerSet};

/// Protocol version for compatibility checking.
/// 
/// Format: `major.minor.patch-qualifier`
/// - Major: Breaking protocol changes
/// - Minor: Backward-compatible additions
/// - Patch: Bug fixes
pub const PROTOCOL_VERSION: &str = "1.0.0-alpha";

/// Minimum cosine similarity for embedding verification.
/// 
/// Validated: cross-platform variance is ~10⁻⁸, well below this threshold.
/// Using 0.9999 provides safety margin while allowing minor floating-point drift.
pub const EMBEDDING_TOLERANCE: f64 = 0.9999;

/// Default HNSW parameters.
/// 
/// Validated on 10k real embeddings from all-MiniLM-L6-v2:
/// - 100% recall at k=10
/// - 45µs average query latency
/// - Good balance of accuracy, speed, and memory
pub mod hnsw {
    /// Graph connectivity parameter (M).
    /// Higher = better recall, more memory.
    pub const CONNECTIVITY: usize = 16;

    /// Construction-time expansion factor (ef_construction).
    /// Higher = better graph quality, slower build.
    pub const EF_CONSTRUCTION: usize = 200;

    /// Search-time expansion factor (ef_search).
    /// Higher = better recall, slower search.
    pub const EF_SEARCH: usize = 50;

    /// Embedding dimension for all-MiniLM-L6-v2.
    pub const EMBEDDING_DIM: usize = 384;

    /// Quantization type for storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Quantization {
        /// Full 32-bit floats.
        F32,
        /// Half-precision 16-bit floats.
        F16,
        /// 8-bit integers (with scaling).
        I8,
    }

    impl Default for Quantization {
        fn default() -> Self {
            Self::F32
        }
    }
}

/// Sync operation result.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Layers that were synced.
    pub synced: Vec<LayerKind>,
    /// Layers that were skipped (already up-to-date).
    pub skipped: Vec<LayerKind>,
    /// Layers that failed to sync.
    pub failed: Vec<(LayerKind, String)>,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
}

impl SyncResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self {
            synced: Vec::new(),
            skipped: Vec::new(),
            failed: Vec::new(),
            bytes_transferred: 0,
        }
    }

    /// Check if sync was fully successful.
    pub fn is_success(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get number of layers synced.
    pub fn synced_count(&self) -> usize {
        self.synced.len()
    }
}

impl Default for SyncResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version_format() {
        assert!(PROTOCOL_VERSION.starts_with("1."));
        assert!(PROTOCOL_VERSION.contains('-') || PROTOCOL_VERSION.chars().all(|c| c.is_ascii_digit() || c == '.'));
    }

    #[test]
    fn test_embedding_tolerance_range() {
        assert!(EMBEDDING_TOLERANCE > 0.99);
        assert!(EMBEDDING_TOLERANCE < 1.0);
    }

    #[test]
    fn test_hnsw_defaults_reasonable() {
        assert!(hnsw::CONNECTIVITY >= 8);
        assert!(hnsw::CONNECTIVITY <= 64);
        assert!(hnsw::EF_CONSTRUCTION >= hnsw::CONNECTIVITY);
        assert!(hnsw::EF_SEARCH >= 10);
        assert_eq!(hnsw::EMBEDDING_DIM, 384); // all-MiniLM-L6-v2
    }

    #[test]
    fn test_sync_result() {
        let mut result = SyncResult::new();
        assert!(result.is_success());

        result.synced.push(LayerKind::Canonical);
        assert!(result.is_success());
        assert_eq!(result.synced_count(), 1);

        result.failed.push((LayerKind::Embedding, "error".into()));
        assert!(!result.is_success());
    }

    #[test]
    fn test_example_from_docs() {
        // Create layer set with canonical data
        let mut layers = LayerSet::new();
        let hash = hash_content(b"document content");
        layers.set_layer(Layer::new(LayerKind::Canonical, hash, 1024));

        // Negotiate sync between peers
        let source = PeerCapability::new("server", CapabilityTier::Full);
        let target = PeerCapability::new("client", CapabilityTier::Lite);
        let plan = negotiate(&source, &target).unwrap();

        // Plan tells us what to ship vs. generate
        assert!(plan.ship_layers.contains(&LayerKind::Canonical));
    }
}
