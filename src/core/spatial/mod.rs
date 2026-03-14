use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::descriptors::{
    CartesianPosition, ChannelSpec, CustomChannelLayout, SpatialDescriptors, SpatialEncoding,
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
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub is_lfe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_m: Option<CartesianPosition>,
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
    #[serde(default)]
    azimuth_deg: Option<f32>,
    #[serde(default)]
    elevation_deg: Option<f32>,
    #[serde(default)]
    is_lfe: bool,
    #[serde(default)]
    position_m: Option<CartesianPosition>,
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

    let mut channels = Vec::with_capacity(raw.channels.len());
    for c in raw.channels {
        let (azimuth_deg, elevation_deg) = resolve_angles(&c)?;
        channels.push(ChannelSpec {
            label: c.label,
            azimuth_deg,
            elevation_deg,
            is_lfe: c.is_lfe,
            position_m: c.position_m,
        });
    }

    let layout = CustomChannelLayout {
        layout_name: raw.layout_name,
        spatial_encoding,
        channels,
    };

    validate_custom_layout(&layout)?;
    Ok(layout)
}

fn resolve_angles(c: &LayoutFileChannel) -> Result<(f32, f32)> {
    match (&c.position_m, c.azimuth_deg, c.elevation_deg) {
        (Some(pos), Some(az), Some(el)) => {
            let (d_az, d_el) = derive_angles_from_position(*pos)?;
            let az_err = angular_abs_delta_deg(az, d_az);
            let el_err = (el - d_el).abs();
            anyhow::ensure!(
                az_err <= 2.0 && el_err <= 2.0,
                "channel '{}' has inconsistent polar/cartesian geometry: provided ({az:.3},{el:.3}) vs derived ({d_az:.3},{d_el:.3})",
                c.label
            );
            Ok((d_az, d_el))
        }
        (Some(pos), _, _) => derive_angles_from_position(*pos),
        (None, Some(az), Some(el)) => Ok((az, el)),
        (None, Some(_), None) | (None, None, Some(_)) => Err(anyhow!(
            "channel '{}' must provide both azimuth_deg and elevation_deg when position_m is omitted",
            c.label
        )),
        (None, None, None) => Err(anyhow!(
            "channel '{}' must provide either position_m or polar angles",
            c.label
        )),
    }
}

fn derive_angles_from_position(pos: CartesianPosition) -> Result<(f32, f32)> {
    let planar = (pos.x * pos.x + pos.y * pos.y).sqrt();
    let radius = (planar * planar + pos.z * pos.z).sqrt();
    anyhow::ensure!(
        radius > 1e-6,
        "position_m cannot be at the capture origin (0,0,0)"
    );

    // Internal convention: +Y is 0 deg azimuth, +X is +90 deg, +Z is elevation.
    let azimuth_deg = pos.x.atan2(pos.y).to_degrees();
    let elevation_deg = pos.z.atan2(planar.max(1e-9)).to_degrees();
    Ok((azimuth_deg, elevation_deg))
}

fn angular_abs_delta_deg(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d < -180.0 {
        d += 360.0;
    }
    d.abs()
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
            ch.azimuth_deg >= -180.0 && ch.azimuth_deg <= 180.0,
            "channel '{label}' azimuth {} out of range [-180, 180]",
            ch.azimuth_deg
        );
        anyhow::ensure!(
            ch.elevation_deg >= -90.0 && ch.elevation_deg <= 90.0,
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
            position_m: c.position_m,
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
            ch.azimuth_deg >= -180.0 && ch.azimuth_deg <= 180.0,
            "channel '{}' azimuth {} out of range [-180, 180]",
            label,
            ch.azimuth_deg
        );
        anyhow::ensure!(
            ch.elevation_deg >= -90.0 && ch.elevation_deg <= 90.0,
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
