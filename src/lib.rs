//! `meta_core` — Shared infrastructure for `~/.meta/` directory management.
//!
//! Provides:
//! - `data_dir` — Locate and create the `~/.meta/` data directory and namespaced files
//! - `lock` — File-based locking with PID staleness detection and retry
//! - `store` — Atomic JSON read/write with lock-protected updates

use std::path::PathBuf;

pub mod data_dir;
pub mod lock;
pub mod store;

/// Default meta data directory name.
const META_DIR_NAME: &str = ".meta";

/// Environment variable to override the meta data directory location.
const META_DATA_DIR_ENV: &str = "META_DATA_DIR";

/// Get the meta data directory path.
/// Respects `META_DATA_DIR` env var, otherwise defaults to `~/.meta/`.
pub fn meta_dir() -> PathBuf {
    if let Ok(override_path) = std::env::var(META_DATA_DIR_ENV) {
        return PathBuf::from(override_path);
    }
    dirs_home().join(META_DIR_NAME)
}

fn dirs_home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp/meta-fallback"))
}

// TODO: Sync protocol layer
// - Layer 0: Canonical data (commits, documents)
// - Layer 1: Embeddings (content-addressed vectors)
// - Layer 2: Indices (HNSW for search)
pub mod sync {
    /// Placeholder for sync protocol implementation
    pub fn protocol_version() -> &'static str {
        "1.0.0-alpha"
    }
}
