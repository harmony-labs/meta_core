//! Peer capability negotiation for sync protocol.
//!
//! Defines what operations a peer can perform, which determines
//! what data needs to be shipped vs. regenerated locally.

use serde::{Deserialize, Serialize};
use super::LayerKind;

/// Individual capability flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capability {
    /// Can generate embeddings from content.
    GenerateEmbeddings,
    
    /// Can build HNSW indices from embeddings.
    BuildIndex,
    
    /// Can ship embeddings to peers.
    ShipEmbeddings,
    
    /// Can ship index data to peers.
    ShipIndex,
    
    /// Can receive and use shipped embeddings.
    ReceiveEmbeddings,
    
    /// Can receive and use shipped indices.
    ReceiveIndex,
    
    /// Can perform semantic search queries.
    SemanticSearch,
}

/// Capability tier for common configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityTier {
    /// Full capabilities: generate, ship, receive everything.
    Full,
    
    /// Lite: can receive and search, but not generate.
    Lite,
    
    /// Thin: receive-only, no local search capability.
    Thin,
}

impl CapabilityTier {
    /// Get the capabilities for this tier.
    pub fn capabilities(&self) -> Vec<Capability> {
        match self {
            CapabilityTier::Full => vec![
                Capability::GenerateEmbeddings,
                Capability::BuildIndex,
                Capability::ShipEmbeddings,
                Capability::ShipIndex,
                Capability::ReceiveEmbeddings,
                Capability::ReceiveIndex,
                Capability::SemanticSearch,
            ],
            CapabilityTier::Lite => vec![
                Capability::ReceiveEmbeddings,
                Capability::ReceiveIndex,
                Capability::SemanticSearch,
            ],
            CapabilityTier::Thin => vec![
                Capability::ReceiveEmbeddings,
                Capability::ReceiveIndex,
            ],
        }
    }
}

/// A peer's declared capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerCapability {
    /// Unique peer identifier.
    pub peer_id: String,
    
    /// Capability tier.
    pub tier: CapabilityTier,
    
    /// Explicit capability overrides (additions or removals).
    #[serde(default)]
    pub overrides: Vec<(Capability, bool)>,
    
    /// Protocol version the peer supports.
    pub protocol_version: String,
}

impl PeerCapability {
    /// Create a new peer capability set.
    pub fn new(peer_id: impl Into<String>, tier: CapabilityTier) -> Self {
        Self {
            peer_id: peer_id.into(),
            tier,
            overrides: Vec::new(),
            protocol_version: super::PROTOCOL_VERSION.to_string(),
        }
    }

    /// Check if this peer has a specific capability.
    pub fn has(&self, cap: Capability) -> bool {
        let base = self.tier.capabilities().contains(&cap);
        
        // Check overrides
        for (c, enabled) in &self.overrides {
            if *c == cap {
                return *enabled;
            }
        }
        
        base
    }

    /// Check if this peer can generate a layer locally.
    pub fn can_generate(&self, kind: LayerKind) -> bool {
        match kind {
            LayerKind::Canonical => false, // Canonical is never "generated"
            LayerKind::Embedding => self.has(Capability::GenerateEmbeddings),
            LayerKind::IndexMeta | LayerKind::IndexData => self.has(Capability::BuildIndex),
        }
    }

    /// Check if this peer can receive a layer.
    pub fn can_receive(&self, kind: LayerKind) -> bool {
        match kind {
            LayerKind::Canonical => true, // Everyone can receive canonical
            LayerKind::Embedding => self.has(Capability::ReceiveEmbeddings),
            LayerKind::IndexMeta | LayerKind::IndexData => self.has(Capability::ReceiveIndex),
        }
    }

    /// Add a capability override.
    pub fn with_override(mut self, cap: Capability, enabled: bool) -> Self {
        self.overrides.push((cap, enabled));
        self
    }
}

/// Result of capability negotiation between two peers.
#[derive(Debug, Clone)]
pub struct NegotiatedCapabilities {
    /// Layers to ship from source to target.
    pub ship_layers: Vec<LayerKind>,
    
    /// Layers target will generate locally.
    pub generate_layers: Vec<LayerKind>,
}

/// Negotiate what to ship between a source and target peer.
pub fn negotiate(_source: &PeerCapability, target: &PeerCapability) -> NegotiatedCapabilities {
    let mut ship = Vec::new();
    let mut generate = Vec::new();

    // Always ship canonical
    ship.push(LayerKind::Canonical);

    // Embeddings: ship if target can receive but can't generate
    if target.can_receive(LayerKind::Embedding) {
        if target.can_generate(LayerKind::Embedding) {
            generate.push(LayerKind::Embedding);
        } else {
            ship.push(LayerKind::Embedding);
        }
    }

    // Index: ship if target can receive but can't build
    if target.can_receive(LayerKind::IndexData) {
        if target.can_generate(LayerKind::IndexData) {
            generate.push(LayerKind::IndexData);
        } else {
            ship.push(LayerKind::IndexMeta);
            ship.push(LayerKind::IndexData);
        }
    }

    NegotiatedCapabilities {
        ship_layers: ship,
        generate_layers: generate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_tier_capabilities() {
        let peer = PeerCapability::new("peer1", CapabilityTier::Full);
        
        assert!(peer.has(Capability::GenerateEmbeddings));
        assert!(peer.has(Capability::BuildIndex));
        assert!(peer.has(Capability::ShipEmbeddings));
        assert!(peer.has(Capability::SemanticSearch));
    }

    #[test]
    fn test_lite_tier_capabilities() {
        let peer = PeerCapability::new("peer1", CapabilityTier::Lite);
        
        assert!(!peer.has(Capability::GenerateEmbeddings));
        assert!(!peer.has(Capability::BuildIndex));
        assert!(peer.has(Capability::ReceiveEmbeddings));
        assert!(peer.has(Capability::SemanticSearch));
    }

    #[test]
    fn test_thin_tier_capabilities() {
        let peer = PeerCapability::new("peer1", CapabilityTier::Thin);
        
        assert!(!peer.has(Capability::GenerateEmbeddings));
        assert!(peer.has(Capability::ReceiveIndex));
        assert!(!peer.has(Capability::SemanticSearch));
    }

    #[test]
    fn test_capability_override() {
        let peer = PeerCapability::new("peer1", CapabilityTier::Lite)
            .with_override(Capability::GenerateEmbeddings, true);
        
        assert!(peer.has(Capability::GenerateEmbeddings));
    }

    #[test]
    fn test_negotiate_full_to_lite() {
        let source = PeerCapability::new("source", CapabilityTier::Full);
        let target = PeerCapability::new("target", CapabilityTier::Lite);
        
        let result = negotiate(&source, &target);
        
        // Should ship everything since target can't generate
        assert!(result.ship_layers.contains(&LayerKind::Canonical));
        assert!(result.ship_layers.contains(&LayerKind::Embedding));
        assert!(result.ship_layers.contains(&LayerKind::IndexData));
        assert!(result.generate_layers.is_empty());
    }

    #[test]
    fn test_negotiate_full_to_full() {
        let source = PeerCapability::new("source", CapabilityTier::Full);
        let target = PeerCapability::new("target", CapabilityTier::Full);
        
        let result = negotiate(&source, &target);
        
        // Only ship canonical, target generates the rest
        assert!(result.ship_layers.contains(&LayerKind::Canonical));
        assert!(!result.ship_layers.contains(&LayerKind::Embedding));
        assert!(result.generate_layers.contains(&LayerKind::Embedding));
        assert!(result.generate_layers.contains(&LayerKind::IndexData));
    }

    #[test]
    fn test_can_generate() {
        let full = PeerCapability::new("full", CapabilityTier::Full);
        let lite = PeerCapability::new("lite", CapabilityTier::Lite);
        
        assert!(full.can_generate(LayerKind::Embedding));
        assert!(!lite.can_generate(LayerKind::Embedding));
        
        // Canonical is never "generated"
        assert!(!full.can_generate(LayerKind::Canonical));
    }
}
