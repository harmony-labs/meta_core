//! File-based locking with PID staleness detection and retry.
//!
//! Uses `O_CREAT | O_EXCL` semantics for atomic lock creation.
//! Writes the current PID into the lock file for stale lock detection.
//! Provides a RAII guard that releases the lock on drop.

use anyhow::{Context, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// RAII guard that releases the lock file on drop.
pub struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl LockGuard {
    /// Get the path of the lock file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Acquire an exclusive lock at the given path.
///
/// Creates the lock file with `O_CREAT | O_EXCL` and writes the current PID.
/// If the lock file already exists:
/// 1. Check if the PID inside is still alive (stale lock detection)
/// 2. If stale, remove and retry
/// 3. If alive, wait `retry_ms` milliseconds and retry up to `max_retries` times
///
/// Returns a `LockGuard` that removes the lock file on drop.
pub fn acquire(lock_path: &Path, max_retries: u32, retry_ms: u64) -> Result<LockGuard> {
    // Ensure parent directory exists
    if let Some(parent) = lock_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create lock directory: {}", parent.display()))?;
        }
    }

    for attempt in 0..=max_retries {
        match try_create_lock(lock_path) {
            Ok(guard) => return Ok(guard),
            Err(_) if attempt < max_retries => {
                // Lock exists — check if stale
                if let Some(stale_pid) = stale_pid(lock_path) {
                    // Double-check: re-read PID to guard against race where
                    // another process acquired the lock between our checks
                    if read_lock_pid(lock_path) == Some(stale_pid) {
                        let _ = fs::remove_file(lock_path);
                    }
                    continue;
                }
                // Lock is held by a live process — wait and retry
                thread::sleep(Duration::from_millis(retry_ms));
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "Failed to acquire lock at {} after {} attempts",
                        lock_path.display(),
                        max_retries + 1
                    )
                });
            }
        }
    }

    anyhow::bail!(
        "Failed to acquire lock at {} after {} retries",
        lock_path.display(),
        max_retries
    )
}

/// Try to create the lock file atomically.
fn try_create_lock(lock_path: &Path) -> Result<LockGuard> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true) // O_CREAT | O_EXCL
        .open(lock_path)
        .with_context(|| format!("Lock file already exists: {}", lock_path.display()))?;

    // Write current PID
    let pid = std::process::id();
    writeln!(file, "{pid}")
        .with_context(|| format!("Failed to write PID to lock file: {}", lock_path.display()))?;

    Ok(LockGuard {
        path: lock_path.to_path_buf(),
    })
}

/// Check if a lock file is stale (the PID inside is dead).
///
/// Returns `true` if:
/// - The lock file doesn't exist
/// - The lock file can't be read
/// - The PID in the lock file is not a running process
pub fn is_stale(lock_path: &Path) -> bool {
    stale_pid(lock_path).is_some()
}

/// Read the PID from a lock file, returning None if unreadable or unparseable.
fn read_lock_pid(lock_path: &Path) -> Option<u32> {
    let content = fs::read_to_string(lock_path).ok()?;
    content.trim().parse().ok()
}

/// If the lock is stale, return the dead PID. Otherwise return None.
fn stale_pid(lock_path: &Path) -> Option<u32> {
    let pid = read_lock_pid(lock_path)?;
    if is_process_alive(pid) {
        None
    } else {
        Some(pid)
    }
}

/// Check if a process with the given PID is alive.
#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    // kill(pid, 0) checks if process exists without sending a signal
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(not(unix))]
fn is_process_alive(_pid: u32) -> bool {
    // On non-Unix, conservatively assume the process is alive
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_and_release() {
        let tmp = tempfile::tempdir().unwrap();
        let lock_path = tmp.path().join("test.lock");

        {
            let guard = acquire(&lock_path, 0, 100).unwrap();
            assert!(lock_path.exists());
            assert_eq!(guard.path(), lock_path);

            // Verify PID was written
            let content = fs::read_to_string(&lock_path).unwrap();
            let pid: u32 = content.trim().parse().unwrap();
            assert_eq!(pid, std::process::id());
        }

        // Guard dropped — lock should be removed
        assert!(!lock_path.exists());
    }

    #[test]
    fn test_acquire_fails_when_locked() {
        let tmp = tempfile::tempdir().unwrap();
        let lock_path = tmp.path().join("test.lock");

        let _guard = acquire(&lock_path, 0, 100).unwrap();

        // Second acquire should fail (0 retries)
        let result = acquire(&lock_path, 0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_stale_lock_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let lock_path = tmp.path().join("stale.lock");

        // Write a fake PID that definitely doesn't exist
        fs::write(&lock_path, "999999999\n").unwrap();

        assert!(is_stale(&lock_path));

        // Write our own PID — should not be stale
        fs::write(&lock_path, format!("{}\n", std::process::id())).unwrap();
        assert!(!is_stale(&lock_path));
    }

    #[test]
    fn test_acquire_recovers_stale_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let lock_path = tmp.path().join("stale.lock");

        // Create a stale lock (non-existent PID)
        fs::write(&lock_path, "999999999\n").unwrap();

        // Should succeed by detecting and removing the stale lock
        let guard = acquire(&lock_path, 1, 10).unwrap();
        assert!(lock_path.exists());
        drop(guard);
    }
}
