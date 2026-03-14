use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn sha256_json<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(sha256_hex(&bytes))
}

pub fn sha256_channels_f32(channels: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((channels.len() as u64).to_le_bytes());
    for ch in channels {
        hasher.update((ch.len() as u64).to_le_bytes());
        for &s in ch {
            hasher.update(s.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}
