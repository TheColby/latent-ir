use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::descriptors::{
    ChannelSpec, CustomChannelLayout, SpatialDescriptors, SpatialEncoding,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMap {
    pub schema_version: String,
    pub layout_name: String,
    pub spatial_encoding: String,
    pub channels: Vec<ChannelMapEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMapEntry {
    pub index: usize,
    pub label: String,
    pub azimuth_deg: i16,
    pub elevation_deg: i16,
    pub is_lfe: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct LayoutFile {
    #[serde(default)]
    schema_version: Option<String>,
    layout_name: String,
    #[serde(default)]
    spatial_encoding: Option<String>,
    channels: Vec<LayoutFileChannel>,
}

#[derive(Debug, Clone, Deserialize)]
struct LayoutFileChannel {
    label: String,
    azimuth_deg: i16,
    elevation_deg: i16,
    #[serde(default)]
    is_lfe: bool,
}

pub fn load_custom_layout_file(path: impl AsRef<Path>) -> Result<CustomChannelLayout> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read custom layout {}", path.display()))?;
    let raw: LayoutFile = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse custom layout JSON {}", path.display()))?;

    if let Some(schema) = raw.schema_version.as_deref() {
        anyhow::ensure!(
            schema == "latent-ir.layout.v1",
            "unsupported layout schema_version '{schema}', expected 'latent-ir.layout.v1'"
        );
    }

    let spatial_encoding = if let Some(enc) = raw.spatial_encoding.as_deref() {
        SpatialEncoding::from_str_name(enc).ok_or_else(|| {
            anyhow!(
                "unsupported spatial_encoding '{enc}' (supported: discrete, ambisonic_foa_ambix)"
            )
        })?
    } else {
        SpatialEncoding::Discrete
    };

    let layout = CustomChannelLayout {
        layout_name: raw.layout_name,
        spatial_encoding,
        channels: raw
            .channels
            .into_iter()
            .map(|c| ChannelSpec {
                label: c.label,
                azimuth_deg: c.azimuth_deg,
                elevation_deg: c.elevation_deg,
                is_lfe: c.is_lfe,
            })
            .collect(),
    };

    validate_custom_layout(&layout)?;
    Ok(layout)
}

pub fn validate_custom_layout(layout: &CustomChannelLayout) -> Result<()> {
    let name = layout.layout_name.trim();
    anyhow::ensure!(!name.is_empty(), "custom layout_name cannot be empty");
    anyhow::ensure!(
        !layout.channels.is_empty(),
        "custom layout must define at least one channel"
    );
    anyhow::ensure!(
        layout.channels.len() <= 128,
        "custom layout has too many channels (max 128)"
    );

    let mut labels = HashSet::new();
    for (idx, ch) in layout.channels.iter().enumerate() {
        let label = ch.label.trim();
        anyhow::ensure!(!label.is_empty(), "channel {idx} has an empty label");
        anyhow::ensure!(
            labels.insert(label.to_ascii_lowercase()),
            "duplicate channel label '{label}'"
        );
        anyhow::ensure!(
            (-180..=180).contains(&ch.azimuth_deg),
            "channel '{label}' azimuth {} out of range [-180, 180]",
            ch.azimuth_deg
        );
        anyhow::ensure!(
            (-90..=90).contains(&ch.elevation_deg),
            "channel '{label}' elevation {} out of range [-90, 90]",
            ch.elevation_deg
        );
    }

    if layout.spatial_encoding == SpatialEncoding::AmbisonicFoaAmbix {
        anyhow::ensure!(
            layout.channels.len() == 4,
            "ambisonic_foa_ambix layout must have exactly 4 channels"
        );
        let expected = ["w", "x", "y", "z"];
        for (idx, exp) in expected.iter().enumerate() {
            let got = layout.channels[idx].label.trim().to_ascii_lowercase();
            anyhow::ensure!(
                got == *exp,
                "ambisonic_foa_ambix channel {} must be '{}' (got '{}')",
                idx,
                exp,
                layout.channels[idx].label
            );
        }
    }

    Ok(())
}

pub fn build_channel_map(spatial: &SpatialDescriptors) -> ChannelMap {
    let channels = spatial
        .resolved_channel_specs()
        .into_iter()
        .enumerate()
        .map(|(index, c)| ChannelMapEntry {
            index,
            label: c.label,
            azimuth_deg: c.azimuth_deg,
            elevation_deg: c.elevation_deg,
            is_lfe: c.is_lfe,
        })
        .collect();

    ChannelMap {
        schema_version: "latent-ir.channel-map.v1".to_string(),
        layout_name: spatial.resolved_layout_name(),
        spatial_encoding: spatial.resolved_spatial_encoding().as_str().to_string(),
        channels,
    }
}

pub fn validate_channel_map(map: &ChannelMap, expected_channels: usize) -> Result<()> {
    anyhow::ensure!(
        map.schema_version == "latent-ir.channel-map.v1",
        "unsupported channel map schema_version '{}'",
        map.schema_version
    );
    anyhow::ensure!(
        map.channels.len() == expected_channels,
        "channel map count {} does not match audio channels {}",
        map.channels.len(),
        expected_channels
    );

    let encoding = SpatialEncoding::from_str_name(&map.spatial_encoding).ok_or_else(|| {
        anyhow!(
            "unsupported channel map spatial_encoding '{}'",
            map.spatial_encoding
        )
    })?;

    let mut labels = HashSet::new();
    for (i, ch) in map.channels.iter().enumerate() {
        anyhow::ensure!(
            ch.index == i,
            "channel map indices must be contiguous and ordered (expected {i}, got {})",
            ch.index
        );
        let label = ch.label.trim();
        anyhow::ensure!(!label.is_empty(), "channel {i} has empty label");
        anyhow::ensure!(
            labels.insert(label.to_ascii_lowercase()),
            "duplicate channel label '{}'",
            label
        );
        anyhow::ensure!(
            (-180..=180).contains(&ch.azimuth_deg),
            "channel '{}' azimuth {} out of range [-180, 180]",
            label,
            ch.azimuth_deg
        );
        anyhow::ensure!(
            (-90..=90).contains(&ch.elevation_deg),
            "channel '{}' elevation {} out of range [-90, 90]",
            label,
            ch.elevation_deg
        );
    }

    if encoding == SpatialEncoding::AmbisonicFoaAmbix {
        anyhow::ensure!(
            map.channels.len() == 4,
            "ambisonic_foa_ambix channel map must have 4 channels"
        );
        let expected = ["w", "x", "y", "z"];
        for (idx, exp) in expected.iter().enumerate() {
            let got = map.channels[idx].label.trim().to_ascii_lowercase();
            anyhow::ensure!(
                got == *exp,
                "ambisonic_foa_ambix channel {} must be '{}' (got '{}')",
                idx,
                exp,
                map.channels[idx].label
            );
        }
    }

    Ok(())
}

pub fn read_channel_map(path: impl AsRef<Path>) -> Result<ChannelMap> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read channel map {}", path.display()))?;
    let map: ChannelMap = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse channel map {}", path.display()))?;
    validate_channel_map(&map, map.channels.len())?;
    Ok(map)
}

pub fn try_read_companion_channel_map(audio_path: impl AsRef<Path>) -> Result<Option<ChannelMap>> {
    let map_path = companion_channel_map_path(audio_path.as_ref());
    if !map_path.exists() {
        return Ok(None);
    }
    let map = read_channel_map(&map_path)?;
    Ok(Some(map))
}

pub fn companion_channel_map_path(audio_path: impl AsRef<Path>) -> PathBuf {
    let mut p = audio_path.as_ref().to_path_buf();
    p.set_extension("channels.json");
    p
}

pub fn ensure_custom_layout_requested(is_custom: bool, has_layout_json: bool) -> Result<()> {
    if is_custom && !has_layout_json {
        bail!("--channels custom requires --layout-json <path>");
    }
    if !is_custom && has_layout_json {
        bail!("--layout-json is only valid with --channels custom");
    }
    Ok(())
}
