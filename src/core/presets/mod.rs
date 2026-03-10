use anyhow::{anyhow, Result};

use crate::core::descriptors::{ChannelFormat, DescriptorSet};

pub fn preset_names() -> Vec<&'static str> {
    vec![
        "intimate_wood_chapel",
        "dark_stone_cathedral",
        "steel_bunker",
        "glass_corridor",
        "frozen_plate",
        "impossible_infinite_tunnel",
    ]
}

pub fn resolve_preset(name: &str) -> Result<DescriptorSet> {
    let mut d = DescriptorSet::default();
    match name {
        "intimate_wood_chapel" => {
            d.time.duration = 2.8;
            d.time.t60 = 1.6;
            d.time.predelay_ms = 9.0;
            d.spectral.brightness = 0.42;
            d.spectral.hf_damping = 0.68;
            d.spectral.lf_bloom = 0.45;
            d.structural.diffusion = 0.58;
            d.structural.early_density = 0.62;
            d.spatial.width = 0.55;
        }
        "dark_stone_cathedral" => {
            d.time.duration = 13.0;
            d.time.t60 = 10.8;
            d.time.predelay_ms = 42.0;
            d.spectral.brightness = 0.2;
            d.spectral.hf_damping = 0.8;
            d.spectral.lf_bloom = 0.75;
            d.structural.diffusion = 0.78;
            d.structural.early_density = 0.35;
            d.structural.late_density = 0.9;
            d.spatial.width = 0.9;
        }
        "steel_bunker" => {
            d.time.duration = 4.2;
            d.time.t60 = 2.8;
            d.time.predelay_ms = 14.0;
            d.spectral.brightness = 0.72;
            d.spectral.hf_damping = 0.25;
            d.spectral.lf_bloom = 0.35;
            d.structural.modal_density = 0.75;
            d.structural.grain = 0.42;
            d.structural.diffusion = 0.44;
            d.spatial.width = 0.62;
        }
        "glass_corridor" => {
            d.time.duration = 6.0;
            d.time.t60 = 4.2;
            d.time.predelay_ms = 24.0;
            d.spectral.brightness = 0.84;
            d.spectral.hf_damping = 0.18;
            d.structural.early_density = 0.45;
            d.structural.diffusion = 0.52;
            d.spatial.width = 0.68;
        }
        "frozen_plate" => {
            d.time.duration = 5.0;
            d.time.t60 = 3.8;
            d.time.predelay_ms = 5.0;
            d.spectral.brightness = 0.9;
            d.spectral.hf_damping = 0.12;
            d.spectral.band_decay_high = 1.15;
            d.structural.diffusion = 0.86;
            d.structural.tail_noise = 0.14;
            d.spatial.width = 0.78;
        }
        "impossible_infinite_tunnel" => {
            d.time.duration = 20.0;
            d.time.t60 = 25.0;
            d.time.predelay_ms = 55.0;
            d.spectral.brightness = 0.35;
            d.spectral.hf_damping = 0.6;
            d.spectral.lf_bloom = 0.82;
            d.structural.diffusion = 0.97;
            d.structural.late_density = 1.0;
            d.spatial.channel_format = ChannelFormat::Stereo;
            d.spatial.width = 1.0;
            d.spatial.decorrelation = 0.95;
        }
        _ => return Err(anyhow!("unknown preset: {name}")),
    }
    d.clamp();
    Ok(d)
}
