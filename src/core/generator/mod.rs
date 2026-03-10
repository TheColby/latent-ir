use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::core::descriptors::{ChannelSpec, DescriptorSet, SpatialEncoding};
use crate::core::perceptual::MacroTrajectory;

#[derive(Debug, Clone)]
pub struct GeneratedIr {
    pub channels: Vec<Vec<f32>>,
}

pub trait IrGenerator {
    fn generate(&self, descriptors: &DescriptorSet, seed: u64) -> Result<GeneratedIr>;
}

#[derive(Debug, Clone)]
pub struct ProceduralIrGenerator {
    sample_rate: u32,
}

impl ProceduralIrGenerator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
}

impl IrGenerator for ProceduralIrGenerator {
    fn generate(&self, descriptors: &DescriptorSet, seed: u64) -> Result<GeneratedIr> {
        let mono = synthesize_mono_core(descriptors, seed, self.sample_rate);
        let mut out = project_channels(
            &mono,
            descriptors,
            seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            self.sample_rate,
        );
        normalize(&mut out);
        Ok(GeneratedIr { channels: out })
    }
}

fn synthesize_mono_core(descriptors: &DescriptorSet, seed: u64, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n = (descriptors.time.duration * sr).round().max(1.0) as usize;
    let mut out = vec![0.0f32; n];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let predelay_samples = (descriptors.time.predelay_ms * 0.001 * sr) as usize;
    if predelay_samples < n {
        out[predelay_samples] = 1.0;
    }

    let early_count = (descriptors.structural.early_density * 120.0).round() as usize + 8;
    for i in 0..early_count {
        let refl = predelay_samples
            + (rng.gen_range(0.0..0.09) * sr * (0.5 + descriptors.structural.diffusion)) as usize;
        if refl >= n {
            continue;
        }

        let gain = 0.8 * (1.0 - (i as f32 / early_count as f32)).powf(1.2);
        let polarity = if rng.gen_bool(0.45) { -1.0 } else { 1.0 };
        out[refl] += gain * polarity;

        let spread = (descriptors.spatial.width * 18.0) as usize;
        let refl_b = (refl + 1 + spread + (i * 5 % 17)).min(n - 1);
        out[refl_b] += gain * 0.33;
    }

    let base_t60 = descriptors.time.t60.max(0.05);
    let low_tau = (base_t60 * descriptors.spectral.band_decay_low) / 6.91;
    let mid_tau = (base_t60 * descriptors.spectral.band_decay_mid) / 6.91;
    let high_tau = (base_t60 * descriptors.spectral.band_decay_high) / 6.91;

    for (i, sample_out) in out.iter_mut().enumerate() {
        let t = i as f32 / sr;
        let env_low = (-t / low_tau.max(0.01)).exp();
        let env_mid = (-t / mid_tau.max(0.01)).exp();
        let env_high = (-t / high_tau.max(0.01)).exp();

        let hf_mix =
            descriptors.spectral.brightness * (1.0 - descriptors.spectral.hf_damping * 0.8);
        let lf_mix = descriptors.spectral.lf_bloom;
        let mid_mix = 1.0 - (0.45 * hf_mix + 0.35 * lf_mix).clamp(0.0, 0.9);

        let density = descriptors.structural.late_density.max(0.01);
        let burst = if rng.gen_bool((0.08 + 0.72 * density) as f64) {
            rng.gen_range(-1.0..1.0)
        } else {
            0.0
        };
        let grain = descriptors.structural.grain;
        let hiss = rng.gen_range(-1.0..1.0) * descriptors.structural.tail_noise * 0.12;
        let sample = burst * (lf_mix * env_low + mid_mix * env_mid + hf_mix * env_high)
            + hiss * env_high * (0.5 + grain);

        *sample_out += sample;
    }

    out
}

fn project_channels(
    mono: &[f32],
    descriptors: &DescriptorSet,
    seed: u64,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let specs = descriptors.spatial.resolved_channel_specs();
    if specs.is_empty() {
        return vec![mono.to_vec()];
    }

    if descriptors.spatial.resolved_spatial_encoding() == SpatialEncoding::AmbisonicFoaAmbix
        && specs.len() == 4
    {
        project_foa_ambix(mono, descriptors, seed, sample_rate)
    } else {
        project_discrete_layout(mono, descriptors, &specs, seed, sample_rate)
    }
}

fn project_discrete_layout(
    mono: &[f32],
    descriptors: &DescriptorSet,
    specs: &[ChannelSpec],
    seed: u64,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let n = mono.len();
    let sr = sample_rate as f32;
    let low = lowpass_onepole(mono, 0.015);
    let mut out = vec![vec![0.0f32; n]; specs.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let src = source_direction(
        descriptors.spatial.asymmetry,
        descriptors.structural.diffusion,
    );
    let width = descriptors.spatial.width;
    let decor = descriptors.spatial.decorrelation;
    let t60 = descriptors.time.t60.max(0.1);

    for (ch_idx, spec) in specs.iter().enumerate() {
        let dir = channel_direction(spec);
        let alignment = dot3(src, dir).clamp(-1.0, 1.0);
        let front_focus = dir[1].max(0.0);
        let spread_gain =
            ((alignment + 1.0) * 0.5 * width + front_focus * (1.0 - width)).clamp(0.0, 1.0);
        let channel_gain = if spec.is_lfe {
            0.5
        } else {
            0.3 + 0.7 * spread_gain
        };

        let delay_ms = decor * (0.35 + ch_idx as f32 * 0.55) + (1.0 - spread_gain) * 0.9;
        let delay_a = (delay_ms * 0.001 * sr).round() as usize;
        let delay_b = delay_a + 1 + ((ch_idx * 7) % 13);
        let blend = (0.82 - 0.5 * decor).clamp(0.2, 0.95);

        for i in 0..n {
            let idx_a = i.saturating_sub(delay_a);
            let idx_b = i.saturating_sub(delay_b);
            let t = i as f32 / sr;
            let env = (-t / (0.35 * t60 + 0.05)).exp();
            let noise = rng.gen_range(-1.0..1.0) * descriptors.structural.tail_noise * 0.02 * env;

            out[ch_idx][i] = if spec.is_lfe {
                low[idx_a] * channel_gain + noise * 0.15
            } else {
                let base = mono[idx_a];
                let smear = mono[idx_b];
                (base * blend + smear * (1.0 - blend)) * channel_gain + noise
            };
        }
    }

    out
}

fn project_foa_ambix(
    mono: &[f32],
    descriptors: &DescriptorSet,
    seed: u64,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let n = mono.len();
    let sr = sample_rate as f32;

    let mut w = vec![0.0f32; n];
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut z = vec![0.0f32; n];

    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xD1B5_4A32_C19F_2A67);
    let src = source_direction(
        descriptors.spatial.asymmetry,
        descriptors.structural.diffusion,
    );
    let width = descriptors.spatial.width;
    let decor = descriptors.spatial.decorrelation;
    let t60 = descriptors.time.t60.max(0.1);

    let delay_x = (decor * 0.0015 * sr).round() as usize + 1;
    let delay_y = (decor * 0.0022 * sr).round() as usize + 2;
    let delay_z = (decor * 0.0031 * sr).round() as usize + 3;

    let dir_gain = 0.25 + 0.65 * width;
    let z_bias = 0.15 + 0.35 * (1.0 - descriptors.structural.diffusion);

    for i in 0..n {
        let t = i as f32 / sr;
        let env = (-t / (0.4 * t60 + 0.03)).exp();

        let b0 = mono[i];
        let bx = mono[i.saturating_sub(delay_x)];
        let by = mono[i.saturating_sub(delay_y)];
        let bz = mono[i.saturating_sub(delay_z)];

        let noise_scale = descriptors.structural.tail_noise * 0.015 * env;
        let nx = rng.gen_range(-1.0..1.0) * noise_scale;
        let ny = rng.gen_range(-1.0..1.0) * noise_scale;
        let nz = rng.gen_range(-1.0..1.0) * noise_scale;

        w[i] = b0 * 0.70710677;
        x[i] = dir_gain * (src[0] * bx + 0.35 * (bx - by)) + nx;
        y[i] = dir_gain * (src[1] * by + 0.35 * (by - bz)) + ny;
        z[i] = dir_gain * (z_bias * bz + 0.25 * (bx - bz)) + nz;
    }

    vec![w, x, y, z]
}

fn source_direction(asymmetry: f32, diffusion: f32) -> [f32; 3] {
    let yaw = asymmetry.clamp(-1.0, 1.0) * std::f32::consts::FRAC_PI_4;
    let mut v = [
        yaw.sin(),
        yaw.cos(),
        0.15 + 0.35 * (1.0 - diffusion.clamp(0.0, 1.0)),
    ];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
    v
}

fn channel_direction(spec: &ChannelSpec) -> [f32; 3] {
    let az = (spec.azimuth_deg as f32).to_radians();
    let el = (spec.elevation_deg as f32).to_radians();

    let mut v = [az.sin() * el.cos(), az.cos() * el.cos(), el.sin()];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
    v
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lowpass_onepole(x: &[f32], alpha: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let mut prev = 0.0;
    for (i, &v) in x.iter().enumerate() {
        prev += alpha * (v - prev);
        y[i] = prev;
    }
    y
}

fn normalize(channels: &mut [Vec<f32>]) {
    let mut peak = 0.0f32;
    for ch in channels.iter() {
        for &s in ch {
            peak = peak.max(s.abs());
        }
    }
    if peak <= 1e-9 {
        return;
    }
    let gain = 0.98 / peak;
    for ch in channels {
        for s in ch {
            *s *= gain;
        }
    }
}

pub fn generate_with_macro_trajectory(
    generator: &ProceduralIrGenerator,
    base: &DescriptorSet,
    trajectory: &MacroTrajectory,
    seed: u64,
) -> Result<GeneratedIr> {
    let segments = 8usize;
    let mut sum: Option<Vec<Vec<f32>>> = None;
    let mut weight_sum: Option<Vec<f32>> = None;

    for s in 0..segments {
        let t = (s as f32 + 0.5) / segments as f32;
        let mut d = base.clone();
        trajectory.sample(t).apply_to(&mut d);
        let part = generator.generate(&d, seed.wrapping_add((s * 7919) as u64))?;

        let n = part.channels.first().map(Vec::len).unwrap_or(0);
        let channels = part.channels.len();
        if sum.is_none() {
            sum = Some(vec![vec![0.0f32; n]; channels]);
            weight_sum = Some(vec![0.0f32; n]);
        }
        let sum_ref = sum.as_mut().expect("sum must exist");
        let w_ref = weight_sum.as_mut().expect("weights must exist");
        let target_n = w_ref.len();

        let center = (t * target_n as f32) as isize;
        let spread = (target_n as f32 / segments as f32).max(1.0);
        for i in 0..target_n {
            let dx = (i as isize - center) as f32 / spread;
            let w = (-0.5 * dx * dx).exp();
            w_ref[i] += w;
            for ch in 0..channels {
                let v = part.channels[ch].get(i).copied().unwrap_or(0.0);
                sum_ref[ch][i] += v * w;
            }
        }
    }

    let mut out = sum.unwrap_or_else(|| vec![vec![]]);
    if let Some(w_ref) = weight_sum {
        for (i, &w) in w_ref.iter().enumerate() {
            let wn = if w <= 1e-9 { 1.0 } else { w };
            for ch in &mut out {
                ch[i] /= wn;
            }
        }
    }
    normalize(&mut out);
    Ok(GeneratedIr { channels: out })
}
