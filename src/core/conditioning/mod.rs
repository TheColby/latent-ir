use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::descriptors::DescriptorSet;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DescriptorDelta {
    pub duration: f32,
    pub predelay_ms: f32,
    pub t60: f32,
    pub edt: f32,
    pub brightness: f32,
    pub hf_damping: f32,
    pub lf_bloom: f32,
    pub spectral_tilt: f32,
    pub band_decay_low: f32,
    pub band_decay_mid: f32,
    pub band_decay_high: f32,
    pub early_density: f32,
    pub late_density: f32,
    pub diffusion: f32,
    pub modal_density: f32,
    pub tail_noise: f32,
    pub grain: f32,
    pub width: f32,
    pub decorrelation: f32,
    pub asymmetry: f32,
}

impl DescriptorDelta {
    pub fn apply_to(&self, d: &mut DescriptorSet, scale: f32) {
        d.time.duration += self.duration * scale;
        d.time.predelay_ms += self.predelay_ms * scale;
        d.time.t60 += self.t60 * scale;
        d.time.edt += self.edt * scale;

        d.spectral.brightness += self.brightness * scale;
        d.spectral.hf_damping += self.hf_damping * scale;
        d.spectral.lf_bloom += self.lf_bloom * scale;
        d.spectral.spectral_tilt += self.spectral_tilt * scale;
        d.spectral.band_decay_low += self.band_decay_low * scale;
        d.spectral.band_decay_mid += self.band_decay_mid * scale;
        d.spectral.band_decay_high += self.band_decay_high * scale;

        d.structural.early_density += self.early_density * scale;
        d.structural.late_density += self.late_density * scale;
        d.structural.diffusion += self.diffusion * scale;
        d.structural.modal_density += self.modal_density * scale;
        d.structural.tail_noise += self.tail_noise * scale;
        d.structural.grain += self.grain * scale;

        d.spatial.width += self.width * scale;
        d.spatial.decorrelation += self.decorrelation * scale;
        d.spatial.asymmetry += self.asymmetry * scale;
        d.clamp();
    }

    pub fn add_inplace(&mut self, other: &Self) {
        self.duration += other.duration;
        self.predelay_ms += other.predelay_ms;
        self.t60 += other.t60;
        self.edt += other.edt;
        self.brightness += other.brightness;
        self.hf_damping += other.hf_damping;
        self.lf_bloom += other.lf_bloom;
        self.spectral_tilt += other.spectral_tilt;
        self.band_decay_low += other.band_decay_low;
        self.band_decay_mid += other.band_decay_mid;
        self.band_decay_high += other.band_decay_high;
        self.early_density += other.early_density;
        self.late_density += other.late_density;
        self.diffusion += other.diffusion;
        self.modal_density += other.modal_density;
        self.tail_noise += other.tail_noise;
        self.grain += other.grain;
        self.width += other.width;
        self.decorrelation += other.decorrelation;
        self.asymmetry += other.asymmetry;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearProjection {
    pub duration: Vec<f32>,
    pub predelay_ms: Vec<f32>,
    pub t60: Vec<f32>,
    pub edt: Vec<f32>,
    pub brightness: Vec<f32>,
    pub hf_damping: Vec<f32>,
    pub lf_bloom: Vec<f32>,
    pub spectral_tilt: Vec<f32>,
    pub band_decay_low: Vec<f32>,
    pub band_decay_mid: Vec<f32>,
    pub band_decay_high: Vec<f32>,
    pub early_density: Vec<f32>,
    pub late_density: Vec<f32>,
    pub diffusion: Vec<f32>,
    pub modal_density: Vec<f32>,
    pub tail_noise: Vec<f32>,
    pub grain: Vec<f32>,
    pub width: Vec<f32>,
    pub decorrelation: Vec<f32>,
    pub asymmetry: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeltaBias {
    pub duration: f32,
    pub predelay_ms: f32,
    pub t60: f32,
    pub edt: f32,
    pub brightness: f32,
    pub hf_damping: f32,
    pub lf_bloom: f32,
    pub spectral_tilt: f32,
    pub band_decay_low: f32,
    pub band_decay_mid: f32,
    pub band_decay_high: f32,
    pub early_density: f32,
    pub late_density: f32,
    pub diffusion: f32,
    pub modal_density: f32,
    pub tail_noise: f32,
    pub grain: f32,
    pub width: f32,
    pub decorrelation: f32,
    pub asymmetry: f32,
}

fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!("dimension mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

impl LinearProjection {
    pub fn project(&self, emb: &[f32], bias: &DeltaBias) -> Result<DescriptorDelta> {
        Ok(DescriptorDelta {
            duration: dot(&self.duration, emb)? + bias.duration,
            predelay_ms: dot(&self.predelay_ms, emb)? + bias.predelay_ms,
            t60: dot(&self.t60, emb)? + bias.t60,
            edt: dot(&self.edt, emb)? + bias.edt,
            brightness: dot(&self.brightness, emb)? + bias.brightness,
            hf_damping: dot(&self.hf_damping, emb)? + bias.hf_damping,
            lf_bloom: dot(&self.lf_bloom, emb)? + bias.lf_bloom,
            spectral_tilt: dot(&self.spectral_tilt, emb)? + bias.spectral_tilt,
            band_decay_low: dot(&self.band_decay_low, emb)? + bias.band_decay_low,
            band_decay_mid: dot(&self.band_decay_mid, emb)? + bias.band_decay_mid,
            band_decay_high: dot(&self.band_decay_high, emb)? + bias.band_decay_high,
            early_density: dot(&self.early_density, emb)? + bias.early_density,
            late_density: dot(&self.late_density, emb)? + bias.late_density,
            diffusion: dot(&self.diffusion, emb)? + bias.diffusion,
            modal_density: dot(&self.modal_density, emb)? + bias.modal_density,
            tail_noise: dot(&self.tail_noise, emb)? + bias.tail_noise,
            grain: dot(&self.grain, emb)? + bias.grain,
            width: dot(&self.width, emb)? + bias.width,
            decorrelation: dot(&self.decorrelation, emb)? + bias.decorrelation,
            asymmetry: dot(&self.asymmetry, emb)? + bias.asymmetry,
        })
    }

    fn validate_dim(&self, expected: usize) -> Result<()> {
        let fields = [
            &self.duration,
            &self.predelay_ms,
            &self.t60,
            &self.edt,
            &self.brightness,
            &self.hf_damping,
            &self.lf_bloom,
            &self.spectral_tilt,
            &self.band_decay_low,
            &self.band_decay_mid,
            &self.band_decay_high,
            &self.early_density,
            &self.late_density,
            &self.diffusion,
            &self.modal_density,
            &self.tail_noise,
            &self.grain,
            &self.width,
            &self.decorrelation,
            &self.asymmetry,
        ];
        if fields.iter().all(|v| v.len() == expected) {
            Ok(())
        } else {
            Err(anyhow!(
                "projection vectors must all have length {expected}"
            ))
        }
    }
}

pub trait TextEncoder {
    fn infer_delta_from_prompt(&self, prompt: &str) -> Result<DescriptorDelta>;
}

pub trait AudioEncoder {
    fn infer_delta_from_audio(
        &self,
        channels: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<DescriptorDelta>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedTextEncoderModel {
    pub model_version: String,
    pub embedding_dim: usize,
    pub token_embeddings: HashMap<String, Vec<f32>>,
    pub unknown_embedding: Vec<f32>,
    pub projection: LinearProjection,
    #[serde(default)]
    pub bias: DeltaBias,
    #[serde(default = "default_scale")]
    pub output_scale: f32,
}

#[derive(Debug, Clone)]
pub struct LearnedTextEncoder {
    model: LearnedTextEncoderModel,
}

impl LearnedTextEncoder {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).with_context(|| {
            format!(
                "failed to read text encoder model {}",
                path.as_ref().display()
            )
        })?;
        let model: LearnedTextEncoderModel = serde_json::from_str(&text)
            .with_context(|| "failed to parse text encoder model JSON")?;
        validate_text_model(&model)?;
        Ok(Self { model })
    }
}

impl TextEncoder for LearnedTextEncoder {
    fn infer_delta_from_prompt(&self, prompt: &str) -> Result<DescriptorDelta> {
        let tokens = tokenize(prompt);
        if tokens.is_empty() {
            return Ok(DescriptorDelta::default());
        }

        let mut emb = vec![0.0f32; self.model.embedding_dim];
        for token in &tokens {
            let token_emb = self
                .model
                .token_embeddings
                .get(token)
                .unwrap_or(&self.model.unknown_embedding);
            for (o, &v) in emb.iter_mut().zip(token_emb.iter()) {
                *o += v;
            }
        }

        let inv = 1.0 / tokens.len() as f32;
        for v in &mut emb {
            *v *= inv;
        }

        let mut delta = self.model.projection.project(&emb, &self.model.bias)?;
        scale_delta(&mut delta, self.model.output_scale);
        Ok(delta)
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn scale_delta(delta: &mut DescriptorDelta, scale: f32) {
    delta.duration *= scale;
    delta.predelay_ms *= scale;
    delta.t60 *= scale;
    delta.edt *= scale;
    delta.brightness *= scale;
    delta.hf_damping *= scale;
    delta.lf_bloom *= scale;
    delta.spectral_tilt *= scale;
    delta.band_decay_low *= scale;
    delta.band_decay_mid *= scale;
    delta.band_decay_high *= scale;
    delta.early_density *= scale;
    delta.late_density *= scale;
    delta.diffusion *= scale;
    delta.modal_density *= scale;
    delta.tail_noise *= scale;
    delta.grain *= scale;
    delta.width *= scale;
    delta.decorrelation *= scale;
    delta.asymmetry *= scale;
}

fn default_scale() -> f32 {
    1.0
}

fn validate_text_model(model: &LearnedTextEncoderModel) -> Result<()> {
    if model.embedding_dim == 0 {
        return Err(anyhow!("embedding_dim must be > 0"));
    }
    if model.unknown_embedding.len() != model.embedding_dim {
        return Err(anyhow!("unknown_embedding length mismatch"));
    }
    if model
        .token_embeddings
        .values()
        .any(|v| v.len() != model.embedding_dim)
    {
        return Err(anyhow!("all token embeddings must match embedding_dim"));
    }
    model.projection.validate_dim(model.embedding_dim)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedAudioEncoderModel {
    pub model_version: String,
    pub feature_names: Vec<String>,
    pub input_mean: Vec<f32>,
    pub input_std: Vec<f32>,
    pub hidden_weights: Vec<Vec<f32>>,
    pub hidden_bias: Vec<f32>,
    pub projection: LinearProjection,
    #[serde(default)]
    pub bias: DeltaBias,
    #[serde(default = "default_scale")]
    pub output_scale: f32,
}

#[derive(Debug, Clone)]
pub struct LearnedAudioEncoder {
    model: LearnedAudioEncoderModel,
}

impl LearnedAudioEncoder {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).with_context(|| {
            format!(
                "failed to read audio encoder model {}",
                path.as_ref().display()
            )
        })?;
        let model: LearnedAudioEncoderModel = serde_json::from_str(&text)
            .with_context(|| "failed to parse audio encoder model JSON")?;
        validate_audio_model(&model)?;
        Ok(Self { model })
    }
}

impl AudioEncoder for LearnedAudioEncoder {
    fn infer_delta_from_audio(
        &self,
        channels: &[Vec<f32>],
        sample_rate: u32,
    ) -> Result<DescriptorDelta> {
        let features = extract_audio_features(channels, sample_rate);
        let in_dim = self.model.feature_names.len();
        if features.len() != in_dim {
            return Err(anyhow!(
                "feature dimension mismatch: extracted={} model={in_dim}",
                features.len()
            ));
        }

        let mut x = vec![0.0f32; in_dim];
        for i in 0..in_dim {
            let std = self.model.input_std[i].abs().max(1e-6);
            x[i] = (features[i] - self.model.input_mean[i]) / std;
        }

        let mut hidden = vec![0.0f32; self.model.hidden_bias.len()];
        for (h, row) in self.model.hidden_weights.iter().enumerate() {
            let z: f32 = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f32>()
                + self.model.hidden_bias[h];
            hidden[h] = z.tanh();
        }

        let mut delta = self.model.projection.project(&hidden, &self.model.bias)?;
        scale_delta(&mut delta, self.model.output_scale);
        Ok(delta)
    }
}

fn validate_audio_model(model: &LearnedAudioEncoderModel) -> Result<()> {
    let in_dim = model.feature_names.len();
    if in_dim == 0 {
        return Err(anyhow!("feature_names cannot be empty"));
    }
    if model.input_mean.len() != in_dim || model.input_std.len() != in_dim {
        return Err(anyhow!(
            "input normalization vectors must match feature count"
        ));
    }
    if model.hidden_weights.is_empty() {
        return Err(anyhow!("hidden_weights cannot be empty"));
    }
    if model.hidden_weights.len() != model.hidden_bias.len() {
        return Err(anyhow!(
            "hidden_weights row count must match hidden_bias length"
        ));
    }
    if model.hidden_weights.iter().any(|row| row.len() != in_dim) {
        return Err(anyhow!("hidden weight rows must match input feature count"));
    }
    model.projection.validate_dim(model.hidden_bias.len())
}

pub fn extract_audio_features(channels: &[Vec<f32>], sample_rate: u32) -> Vec<f32> {
    let mono = downmix(channels);
    if mono.is_empty() {
        return vec![0.0; 10];
    }

    let n = mono.len();
    let sr = sample_rate as f32;
    let duration_s = n as f32 / sr;
    let peak = mono.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let rms = (mono.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();

    let zcr = mono
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count() as f32
        / n.max(1) as f32;

    let predelay_idx = mono.iter().position(|x| x.abs() >= peak * 0.1).unwrap_or(0) as f32;
    let predelay_norm = (predelay_idx / sr).clamp(0.0, 1.0);

    let split_a = (0.08 * sr) as usize;
    let split = split_a.min(n);
    let early = mono[..split].iter().map(|x| x * x).sum::<f32>();
    let late = mono[split..].iter().map(|x| x * x).sum::<f32>();
    let total = (early + late).max(1e-12);
    let early_ratio = early / total;

    let (low_ratio, mid_ratio, high_ratio) = band_energy_ratios(&mono);

    vec![
        duration_s,
        peak,
        rms,
        zcr,
        predelay_norm,
        early_ratio,
        low_ratio,
        mid_ratio,
        high_ratio,
        spectral_centroid_norm(&mono),
    ]
}

fn downmix(channels: &[Vec<f32>]) -> Vec<f32> {
    match channels {
        [] => vec![],
        [mono] => mono.clone(),
        many => {
            let n = many[0].len();
            let mut out = vec![0.0f32; n];
            let inv = 1.0 / many.len() as f32;
            for ch in many {
                for (o, &s) in out.iter_mut().zip(ch.iter()) {
                    *o += s * inv;
                }
            }
            out
        }
    }
}

fn spectral_centroid_norm(x: &[f32]) -> f32 {
    if x.len() < 8 {
        return 0.0;
    }
    let n = x.len();
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &v) in x.iter().enumerate() {
        let w = v.abs();
        let f = i as f32 / n as f32;
        num += f * w;
        den += w;
    }
    if den <= 1e-9 {
        0.0
    } else {
        (num / den).clamp(0.0, 1.0)
    }
}

fn band_energy_ratios(x: &[f32]) -> (f32, f32, f32) {
    let low = lowpass(x, 0.03);
    let high = highpass(x, 0.2);
    let mid: Vec<f32> = x
        .iter()
        .zip(low.iter())
        .zip(high.iter())
        .map(|((&s, &l), &h)| s - l - h)
        .collect();

    let el = low.iter().map(|v| v * v).sum::<f32>();
    let em = mid.iter().map(|v| v * v).sum::<f32>();
    let eh = high.iter().map(|v| v * v).sum::<f32>();
    let sum = (el + em + eh).max(1e-12);
    (el / sum, em / sum, eh / sum)
}

fn lowpass(x: &[f32], a: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let mut s = 0.0;
    for (i, &v) in x.iter().enumerate() {
        s += a * (v - s);
        y[i] = s;
    }
    y
}

fn highpass(x: &[f32], a: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let mut lp = 0.0;
    for (i, &v) in x.iter().enumerate() {
        lp += a * (v - lp);
        y[i] = v - lp;
    }
    y
}
