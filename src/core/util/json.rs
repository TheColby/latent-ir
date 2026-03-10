use anyhow::Result;
use serde::Serialize;
use std::path::Path;

pub fn write_pretty_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text)?;
    Ok(())
}
