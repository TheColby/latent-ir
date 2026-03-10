use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::core::descriptors::DescriptorSet;
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
        let sr = self.sample_rate as f32;
        let n = (descriptors.time.duration * sr).round().max(1.0) as usize;
        let channels = descriptors.spatial.channel_format.channels();
        let mut out = vec![vec![0.0f32; n]; channels];
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let predelay_samples = (descriptors.time.predelay_ms * 0.001 * sr) as usize;
        for ch in &mut out {
            if predelay_samples < ch.len() {
                ch[predelay_samples] = 1.0;
            }
        }

        let early_count = (descriptors.structural.early_density * 120.0).round() as usize + 8;
        for i in 0..early_count {
            let refl = predelay_samples
                + (rng.gen_range(0.0..0.09) * sr * (0.5 + descriptors.structural.diffusion))
                    as usize;
            if refl >= n {
                continue;
            }
            let gain = 0.8 * (1.0 - (i as f32 / early_count as f32)).powf(1.2);
            let polarity = if rng.gen_bool(0.45) { -1.0 } else { 1.0 };
            out[0][refl] += gain * polarity;
            if channels > 1 {
                let spread = (descriptors.spatial.width * 12.0) as usize;
                let other = (refl + (i * 3 + spread)).min(n - 1);
                out[1][other] += gain * (1.0 - descriptors.spatial.asymmetry).max(0.2);
            }
        }

        let base_t60 = descriptors.time.t60.max(0.05);
        let low_tau = (base_t60 * descriptors.spectral.band_decay_low) / 6.91;
        let mid_tau = (base_t60 * descriptors.spectral.band_decay_mid) / 6.91;
        let high_tau = (base_t60 * descriptors.spectral.band_decay_high) / 6.91;

        for i in 0..n {
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

            out[0][i] += sample;
            if channels > 1 {
                let decor = descriptors.spatial.decorrelation;
                let jitter =
                    ((rng.gen_range(-1.0..1.0) * decor * 3.0) as i32).unsigned_abs() as usize;
                let idx = i.saturating_sub(jitter);
                out[1][i] += out[0][idx] * (0.85 + 0.15 * descriptors.spatial.width);
            }
        }

        normalize(&mut out);
        Ok(GeneratedIr { channels: out })
    }
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
