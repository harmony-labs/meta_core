//! Peer capability negotiation for sync protocol.
//!
//! Defines what operations a peer can perform, which determines
//! what data needs to be shipped vs. regenerated locally.

use super::LayerKind;
use serde::{Deserialize, Serialize};

/// Individual capability flags.
/// 
/// Uses a compact bitflag representation internally for efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl Capability {
    /// All capabilities.
    pub const ALL: [Capability; 7] = [
        Capability::GenerateEmbeddings,
        Capability::BuildIndex,
        Capability::ShipEmbeddings,
        Capability::ShipIndex,
        Capability::ReceiveEmbeddings,
        Capability::ReceiveIndex,
        Capability::SemanticSearch,
    ];

    /// Convert to a bit position for compact storage.
    #[inline]
    const fn bit_pos(self) -> u8 {
        match self {
            Capability::GenerateEmbeddings => 0,
            Capability::BuildIndex => 1,
            Capability::ShipEmbeddings => 2,
            Capability::ShipIndex => 3,
            Capability::ReceiveEmbeddings => 4,
            Capability::ReceiveIndex => 5,
            Capability::SemanticSearch => 6,
        }
    }

    /// Convert to a bitmask.
    #[inline]
    const fn mask(self) -> u8 {
        1 << self.bit_pos()
    }
}

/// Capability tier for common configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityTier {
    /// Full capabilities: generate, ship, receive everything.
    Full,

    /// Lite: can receive and search, but not generate.
    Lite,

    /// Thin: receive-only, no local search capability.
    Thin,
}

impl CapabilityTier {
    /// Get the capability bitmask for this tier.
    #[inline]
    const fn bitmask(self) -> u8 {
        match self {
            CapabilityTier::Full => 0b1111111, // All 7 capabilities
            CapabilityTier::Lite => 0b1110000, // Receive + Search
            CapabilityTier::Thin => 0b0110000, // Receive only
        }
    }

    /// Get the capabilities for this tier as a set.
    pub fn capabilities(self) -> CapabilitySet {
        CapabilitySet(self.bitmask())
    }
}

/// A set of capabilities stored as a bitmask.
/// 
/// Compact (1 byte) and efficient for set operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    /// Empty capability set.
    pub const EMPTY: Self = Self(0);

    /// Full capability set.
    pub const FULL: Self = Self(0b1111111);

    /// Create from a tier.
    #[inline]
    pub const fn from_tier(tier: CapabilityTier) -> Self {
        Self(tier.bitmask())
    }

    /// Add a capability.
    #[inline]
    pub fn insert(&mut self, cap: Capability) {
        self.0 |= cap.mask();
    }

    /// Remove a capability.
    #[inline]
    pub fn remove(&mut self, cap: Capability) {
        self.0 &= !cap.mask();
    }

    /// Check if a capability is present.
    #[inline]
    pub const fn contains(self, cap: Capability) -> bool {
        (self.0 & cap.mask()) != 0
    }

    /// Union of two capability sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection of two capability sets.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Count capabilities.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterate over present capabilities.
    pub fn iter(self) -> impl Iterator<Item = Capability> {
        Capability::ALL
            .into_iter()
            .filter(move |&c| self.contains(c))
    }
}

impl Serialize for CapabilitySet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as array of capability names
        let caps: Vec<_> = self.iter().collect();
        caps.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CapabilitySet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let caps: Vec<Capability> = Vec::deserialize(deserializer)?;
        let mut set = CapabilitySet::EMPTY;
        for cap in caps {
            set.insert(cap);
        }
        Ok(set)
    }
}

/// A peer's declared capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerCapability {
    /// Unique peer identifier.
    pub peer_id: String,

    /// Base capability tier.
    pub tier: CapabilityTier,

    /// Effective capabilities (tier + overrides).
    #[serde(default)]
    capabilities: CapabilitySet,

    /// Protocol version the peer supports.
    pub protocol_version: String,
}

impl PeerCapability {
    /// Create a new peer capability set.
    pub fn new(peer_id: impl Into<String>, tier: CapabilityTier) -> Self {
        Self {
            peer_id: peer_id.into(),
            tier,
            capabilities: tier.capabilities(),
            protocol_version: super::PROTOCOL_VERSION.to_string(),
        }
    }

    /// Check if this peer has a specific capability.
    #[inline]
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Add a capability.
    pub fn add_capability(&mut self, cap: Capability) {
        self.capabilities.insert(cap);
    }

    /// Remove a capability.
    pub fn remove_capability(&mut self, cap: Capability) {
        self.capabilities.remove(cap);
    }

    /// Builder: add a capability.
    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.add_capability(cap);
        self
    }

    /// Builder: remove a capability.
    pub fn without_capability(mut self, cap: Capability) -> Self {
        self.remove_capability(cap);
        self
    }

    /// Get the effective capability set.
    #[inline]
    pub fn capabilities(&self) -> CapabilitySet {
        self.capabilities
    }

    /// Check if this peer can generate a layer locally.
    #[inline]
    pub fn can_generate(&self, kind: LayerKind) -> bool {
        match kind {
            LayerKind::Canonical => false, // Canonical is never "generated"
            LayerKind::Embedding => self.has(Capability::GenerateEmbeddings),
            LayerKind::IndexMeta | LayerKind::IndexData => self.has(Capability::BuildIndex),
        }
    }

    /// Check if this peer can receive a layer.
    #[inline]
    pub fn can_receive(&self, kind: LayerKind) -> bool {
        match kind {
            LayerKind::Canonical => true, // Everyone can receive canonical
            LayerKind::Embedding => self.has(Capability::ReceiveEmbeddings),
            LayerKind::IndexMeta | LayerKind::IndexData => self.has(Capability::ReceiveIndex),
        }
    }

    /// Check if this peer can ship a layer.
    #[inline]
    pub fn can_ship(&self, kind: LayerKind) -> bool {
        match kind {
            LayerKind::Canonical => true, // Everyone can ship canonical
            LayerKind::Embedding => self.has(Capability::ShipEmbeddings),
            LayerKind::IndexMeta | LayerKind::IndexData => self.has(Capability::ShipIndex),
        }
    }
}

/// Error during capability negotiation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum NegotiationError {
    /// A required layer cannot be provided by either peer.
    #[error("layer {layer:?} unavailable: source cannot ship and target cannot generate")]
    LayerUnavailable { layer: LayerKind },
}

/// Result of capability negotiation between two peers.
#[derive(Debug, Clone, Default)]
pub struct SyncPlan {
    /// Layers to ship from source to target.
    pub ship_layers: Vec<LayerKind>,

    /// Layers target will generate locally.
    pub generate_layers: Vec<LayerKind>,

    /// Layers that cannot be synced (neither ship nor generate).
    /// Only populated when using `negotiate_permissive`.
    pub unavailable_layers: Vec<LayerKind>,
}

impl SyncPlan {
    /// Check if the plan covers all derived layers.
    pub fn is_complete(&self) -> bool {
        self.unavailable_layers.is_empty()
    }

    /// Get total number of layers in the plan.
    pub fn total_layers(&self) -> usize {
        self.ship_layers.len() + self.generate_layers.len()
    }
}

/// Negotiate what to ship between a source and target peer.
/// 
/// Returns a plan specifying what layers to ship and what to generate locally.
/// Fails if any required layer cannot be provided.
///
/// Use `negotiate_permissive` if you want to allow incomplete syncs.
pub fn negotiate(source: &PeerCapability, target: &PeerCapability) -> Result<SyncPlan, NegotiationError> {
    let plan = negotiate_permissive(source, target);
    
    // Fail if any layer is unavailable
    if let Some(&layer) = plan.unavailable_layers.first() {
        return Err(NegotiationError::LayerUnavailable { layer });
    }
    
    Ok(plan)
}

/// Negotiate sync plan, allowing incomplete syncs.
/// 
/// Unlike `negotiate`, this returns unavailable layers in the plan
/// instead of failing. Use this when partial sync is acceptable.
pub fn negotiate_permissive(source: &PeerCapability, target: &PeerCapability) -> SyncPlan {
    let mut plan = SyncPlan::default();

    // Always ship canonical
    plan.ship_layers.push(LayerKind::Canonical);

    // For each derived layer, decide: ship, generate, or unavailable
    for kind in [LayerKind::Embedding, LayerKind::IndexData] {
        if !target.can_receive(kind) {
            // Target can't use this layer at all - skip silently
            continue;
        }

        if target.can_generate(kind) {
            // Target can generate locally - more efficient
            plan.generate_layers.push(kind);
        } else if source.can_ship(kind) {
            // Source can ship, target will receive
            plan.ship_layers.push(kind);
            // Also ship metadata for indices
            if kind == LayerKind::IndexData {
                plan.ship_layers.push(LayerKind::IndexMeta);
            }
        } else {
            // Neither can provide this layer
            plan.unavailable_layers.push(kind);
        }
    }

    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_bitmask() {
        assert_eq!(Capability::GenerateEmbeddings.mask(), 0b0000001);
        assert_eq!(Capability::SemanticSearch.mask(), 0b1000000);
    }

    #[test]
    fn test_capability_set_operations() {
        let mut set = CapabilitySet::EMPTY;
        assert!(!set.contains(Capability::GenerateEmbeddings));

        set.insert(Capability::GenerateEmbeddings);
        assert!(set.contains(Capability::GenerateEmbeddings));
        assert_eq!(set.count(), 1);

        set.remove(Capability::GenerateEmbeddings);
        assert!(!set.contains(Capability::GenerateEmbeddings));
    }

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
            .with_capability(Capability::GenerateEmbeddings);

        assert!(peer.has(Capability::GenerateEmbeddings));
    }

    #[test]
    fn test_negotiate_full_to_lite() {
        let source = PeerCapability::new("source", CapabilityTier::Full);
        let target = PeerCapability::new("target", CapabilityTier::Lite);

        let plan = negotiate(&source, &target).unwrap();

        // Should ship everything since target can't generate
        assert!(plan.ship_layers.contains(&LayerKind::Canonical));
        assert!(plan.ship_layers.contains(&LayerKind::Embedding));
        assert!(plan.ship_layers.contains(&LayerKind::IndexData));
        assert!(plan.generate_layers.is_empty());
        assert!(plan.is_complete());
    }

    #[test]
    fn test_negotiate_full_to_full() {
        let source = PeerCapability::new("source", CapabilityTier::Full);
        let target = PeerCapability::new("target", CapabilityTier::Full);

        let plan = negotiate(&source, &target).unwrap();

        // Only ship canonical, target generates the rest
        assert!(plan.ship_layers.contains(&LayerKind::Canonical));
        assert!(!plan.ship_layers.contains(&LayerKind::Embedding));
        assert!(plan.generate_layers.contains(&LayerKind::Embedding));
        assert!(plan.generate_layers.contains(&LayerKind::IndexData));
    }

    #[test]
    fn test_negotiate_impossible_fails() {
        // Source can't ship, target can't generate
        let source = PeerCapability::new("source", CapabilityTier::Thin);
        let target = PeerCapability::new("target", CapabilityTier::Lite);

        let result = negotiate(&source, &target);
        assert!(result.is_err());
        
        // Permissive version should succeed with unavailable layers
        let plan = negotiate_permissive(&source, &target);
        assert!(!plan.is_complete());
        assert!(!plan.unavailable_layers.is_empty());
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

    #[test]
    fn test_capability_set_serde() {
        let set = CapabilityTier::Full.capabilities();
        let json = serde_json::to_string(&set).unwrap();
        let parsed: CapabilitySet = serde_json::from_str(&json).unwrap();
        assert_eq!(set, parsed);
    }
}
