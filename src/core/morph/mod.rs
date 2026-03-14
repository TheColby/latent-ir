use crate::core::descriptors::DescriptorSet;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Default, Clone)]
pub struct IrMorpher;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaKeyframe {
    pub t: f32,
    pub alpha: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaTrajectory {
    pub keyframes: Vec<AlphaKeyframe>,
}

impl IrMorpher {
    pub fn morph(&self, a: &[Vec<f32>], b: &[Vec<f32>], alpha: f32) -> Vec<Vec<f32>> {
        let alpha = alpha.clamp(0.0, 1.0);
        let channels = a.len().max(b.len());
        let len = a
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(b.iter().map(Vec::len).max().unwrap_or(0));
        let mut out = vec![vec![0.0; len]; channels];

        for ch in 0..channels {
            let a_ch = a.get(ch).or_else(|| a.first());
            let b_ch = b.get(ch).or_else(|| b.first());
            for i in 0..len {
                let av = a_ch.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                let bv = b_ch.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                out[ch][i] = av * (1.0 - alpha) + bv * alpha;
            }
        }

        normalize(&mut out);
        out
    }

    pub fn morph_descriptors(
        &self,
        a: &DescriptorSet,
        b: &DescriptorSet,
        alpha: f32,
    ) -> DescriptorSet {
        let alpha = alpha.clamp(0.0, 1.0);
        let lerp = |x: f32, y: f32| x * (1.0 - alpha) + y * alpha;

        let mut out = a.clone();
        out.time.duration = lerp(a.time.duration, b.time.duration);
        out.time.predelay_ms = lerp(a.time.predelay_ms, b.time.predelay_ms);
        out.time.t60 = lerp(a.time.t60, b.time.t60);
        out.time.edt = lerp(a.time.edt, b.time.edt);
        out.time.attack_gap_ms = lerp(a.time.attack_gap_ms, b.time.attack_gap_ms);

        out.spectral.brightness = lerp(a.spectral.brightness, b.spectral.brightness);
        out.spectral.hf_damping = lerp(a.spectral.hf_damping, b.spectral.hf_damping);
        out.spectral.lf_bloom = lerp(a.spectral.lf_bloom, b.spectral.lf_bloom);
        out.spectral.spectral_tilt = lerp(a.spectral.spectral_tilt, b.spectral.spectral_tilt);
        out.spectral.band_decay_low = lerp(a.spectral.band_decay_low, b.spectral.band_decay_low);
        out.spectral.band_decay_mid = lerp(a.spectral.band_decay_mid, b.spectral.band_decay_mid);
        out.spectral.band_decay_high = lerp(a.spectral.band_decay_high, b.spectral.band_decay_high);

        out.structural.early_density = lerp(a.structural.early_density, b.structural.early_density);
        out.structural.late_density = lerp(a.structural.late_density, b.structural.late_density);
        out.structural.diffusion = lerp(a.structural.diffusion, b.structural.diffusion);
        out.structural.modal_density = lerp(a.structural.modal_density, b.structural.modal_density);
        out.structural.tail_noise = lerp(a.structural.tail_noise, b.structural.tail_noise);
        out.structural.grain = lerp(a.structural.grain, b.structural.grain);

        out.spatial.width = lerp(a.spatial.width, b.spatial.width);
        out.spatial.decorrelation = lerp(a.spatial.decorrelation, b.spatial.decorrelation);
        out.spatial.asymmetry = lerp(a.spatial.asymmetry, b.spatial.asymmetry);
        out
    }

    pub fn morph_with_trajectory(
        &self,
        a: &[Vec<f32>],
        b: &[Vec<f32>],
        trajectory: &AlphaTrajectory,
    ) -> Vec<Vec<f32>> {
        let channels = a.len().max(b.len());
        let len = a
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(b.iter().map(Vec::len).max().unwrap_or(0));
        let mut out = vec![vec![0.0; len]; channels];

        for ch in 0..channels {
            let a_ch = a.get(ch).or_else(|| a.first());
            let b_ch = b.get(ch).or_else(|| b.first());
            for i in 0..len {
                let phase = if len <= 1 {
                    0.0
                } else {
                    i as f32 / (len - 1) as f32
                };
                let alpha = trajectory.sample(phase);
                let av = a_ch.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                let bv = b_ch.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                out[ch][i] = av * (1.0 - alpha) + bv * alpha;
            }
        }

        normalize(&mut out);
        out
    }
}

impl AlphaTrajectory {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read trajectory {}", path.display()))?;
        let mut traj: AlphaTrajectory =
            serde_json::from_str(&text).with_context(|| "failed to parse alpha trajectory JSON")?;
        traj.normalize_keyframes()?;
        Ok(traj)
    }

    pub fn sample(&self, t: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.5;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.keyframes[0].t {
            return self.keyframes[0].alpha;
        }
        for w in self.keyframes.windows(2) {
            let a = &w[0];
            let b = &w[1];
            if t >= a.t && t <= b.t {
                let denom = (b.t - a.t).abs().max(1e-9);
                let x = (t - a.t) / denom;
                return a.alpha * (1.0 - x) + b.alpha * x;
            }
        }
        self.keyframes[self.keyframes.len() - 1].alpha
    }

    fn normalize_keyframes(&mut self) -> Result<()> {
        anyhow::ensure!(
            !self.keyframes.is_empty(),
            "alpha trajectory requires at least one keyframe"
        );
        for k in &mut self.keyframes {
            anyhow::ensure!(
                k.t.is_finite() && k.alpha.is_finite(),
                "alpha trajectory values must be finite"
            );
            k.t = k.t.clamp(0.0, 1.0);
            k.alpha = k.alpha.clamp(0.0, 1.0);
        }
        self.keyframes.sort_by(|a, b| a.t.total_cmp(&b.t));
        Ok(())
    }
}

fn normalize(channels: &mut [Vec<f32>]) {
    let peak = channels
        .iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f32, |m, &s| m.max(s.abs()));
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
