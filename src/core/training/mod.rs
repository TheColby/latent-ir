use std::cmp::Reverse;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::conditioning::{
    extract_audio_features, DeltaBias, LearnedAudioEncoderModel, LearnedTextEncoderModel,
    LinearProjection,
};
use crate::core::descriptors::DescriptorSet;
use crate::core::util;

const OUTPUT_DIM: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextTrainSample {
    pub prompt: String,
    pub descriptor: DescriptorSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTrainSample {
    pub audio_path: String,
    pub descriptor: DescriptorSet,
}

#[derive(Debug, Clone)]
pub struct TextTrainingConfig {
    pub max_vocab: usize,
    pub min_count: usize,
    pub epochs: usize,
    pub lr: f32,
    pub l2: f32,
}

#[derive(Debug, Clone)]
pub struct AudioTrainingConfig {
    pub epochs: usize,
    pub lr: f32,
    pub l2: f32,
}

pub fn train_text_encoder(
    dataset_path: &Path,
    cfg: &TextTrainingConfig,
) -> Result<LearnedTextEncoderModel> {
    let text = std::fs::read_to_string(dataset_path)
        .with_context(|| format!("failed to read dataset {}", dataset_path.display()))?;
    let samples: Vec<TextTrainSample> = serde_json::from_str(&text)
        .with_context(|| "failed to parse text training dataset JSON")?;
    if samples.is_empty() {
        return Err(anyhow!("text dataset is empty"));
    }

    let vocab = build_vocab(&samples, cfg.max_vocab, cfg.min_count);
    if vocab.is_empty() {
        return Err(anyhow!("text vocabulary is empty after filtering"));
    }

    let index: HashMap<String, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i))
        .collect();

    let x: Vec<Vec<f32>> = samples
        .iter()
        .map(|s| prompt_features(&s.prompt, &index, vocab.len()))
        .collect();
    let y: Vec<[f32; OUTPUT_DIM]> = samples
        .iter()
        .map(|s| descriptor_delta_vector(&s.descriptor))
        .collect();

    let (w, b) = train_linear(&x, &y, cfg.epochs, cfg.lr, cfg.l2);

    let token_embeddings = vocab
        .iter()
        .enumerate()
        .map(|(i, token)| {
            let mut basis = vec![0.0f32; vocab.len()];
            basis[i] = 1.0;
            (token.clone(), basis)
        })
        .collect();

    Ok(LearnedTextEncoderModel {
        model_version: "latent-ir.text-encoder.v1".to_string(),
        embedding_dim: vocab.len(),
        token_embeddings,
        unknown_embedding: vec![0.0; vocab.len()],
        projection: projection_from_weights(&w),
        bias: bias_from_vector(&b),
        output_scale: 1.0,
    })
}

pub fn train_audio_encoder(
    dataset_path: &Path,
    cfg: &AudioTrainingConfig,
) -> Result<LearnedAudioEncoderModel> {
    let text = std::fs::read_to_string(dataset_path)
        .with_context(|| format!("failed to read dataset {}", dataset_path.display()))?;
    let samples: Vec<AudioTrainSample> = serde_json::from_str(&text)
        .with_context(|| "failed to parse audio training dataset JSON")?;
    if samples.is_empty() {
        return Err(anyhow!("audio dataset is empty"));
    }

    let dataset_root = dataset_path.parent().unwrap_or_else(|| Path::new("."));
    let mut raw_x = Vec::with_capacity(samples.len());
    let mut y = Vec::with_capacity(samples.len());

    for sample in &samples {
        let audio_path = resolve_path(dataset_root, Path::new(&sample.audio_path));
        let wav = util::audio::read_wav_f32(&audio_path)
            .with_context(|| format!("failed to read {}", audio_path.display()))?;
        raw_x.push(extract_audio_features(&wav.channels, wav.sample_rate));
        y.push(descriptor_delta_vector(&sample.descriptor));
    }

    let feature_names = vec![
        "duration_s".to_string(),
        "peak".to_string(),
        "rms".to_string(),
        "zcr".to_string(),
        "predelay_norm".to_string(),
        "early_ratio".to_string(),
        "low_ratio".to_string(),
        "mid_ratio".to_string(),
        "high_ratio".to_string(),
        "centroid_norm".to_string(),
    ];

    let (input_mean, input_std) = mean_std_columns(&raw_x);
    let x_norm = normalize_rows(&raw_x, &input_mean, &input_std);
    let hidden: Vec<Vec<f32>> = x_norm
        .iter()
        .map(|row| row.iter().map(|v| v.tanh()).collect())
        .collect();

    let (w, b) = train_linear(&hidden, &y, cfg.epochs, cfg.lr, cfg.l2);

    let hidden_dim = feature_names.len();
    let mut hidden_weights = vec![vec![0.0f32; hidden_dim]; hidden_dim];
    for (i, row) in hidden_weights.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    Ok(LearnedAudioEncoderModel {
        model_version: "latent-ir.audio-encoder.v1".to_string(),
        feature_names,
        input_mean,
        input_std,
        hidden_weights,
        hidden_bias: vec![0.0; hidden_dim],
        projection: projection_from_weights(&w),
        bias: bias_from_vector(&b),
        output_scale: 1.0,
    })
}

fn resolve_path(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

fn build_vocab(samples: &[TextTrainSample], max_vocab: usize, min_count: usize) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for sample in samples {
        for token in tokenize(&sample.prompt) {
            *counts.entry(token).or_insert(0) += 1;
        }
    }

    let mut items: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(_, c)| *c >= min_count)
        .collect();
    items.sort_by_key(|(_, c)| Reverse(*c));
    items.truncate(max_vocab);
    items.into_iter().map(|(t, _)| t).collect()
}

fn prompt_features(prompt: &str, index: &HashMap<String, usize>, dim: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; dim];
    let tokens = tokenize(prompt);
    if tokens.is_empty() {
        return x;
    }
    for tok in tokens.iter() {
        if let Some(&i) = index.get(tok) {
            x[i] += 1.0;
        }
    }
    let inv = 1.0 / tokens.len() as f32;
    for v in &mut x {
        *v *= inv;
    }
    x
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn descriptor_delta_vector(d: &DescriptorSet) -> [f32; OUTPUT_DIM] {
    let b = DescriptorSet::default();
    [
        d.time.duration - b.time.duration,
        d.time.predelay_ms - b.time.predelay_ms,
        d.time.t60 - b.time.t60,
        d.time.edt - b.time.edt,
        d.spectral.brightness - b.spectral.brightness,
        d.spectral.hf_damping - b.spectral.hf_damping,
        d.spectral.lf_bloom - b.spectral.lf_bloom,
        d.spectral.spectral_tilt - b.spectral.spectral_tilt,
        d.spectral.band_decay_low - b.spectral.band_decay_low,
        d.spectral.band_decay_mid - b.spectral.band_decay_mid,
        d.spectral.band_decay_high - b.spectral.band_decay_high,
        d.structural.early_density - b.structural.early_density,
        d.structural.late_density - b.structural.late_density,
        d.structural.diffusion - b.structural.diffusion,
        d.structural.modal_density - b.structural.modal_density,
        d.structural.tail_noise - b.structural.tail_noise,
        d.structural.grain - b.structural.grain,
        d.spatial.width - b.spatial.width,
        d.spatial.decorrelation - b.spatial.decorrelation,
        d.spatial.asymmetry - b.spatial.asymmetry,
    ]
}

fn train_linear(
    x: &[Vec<f32>],
    y: &[[f32; OUTPUT_DIM]],
    epochs: usize,
    lr: f32,
    l2: f32,
) -> (Vec<Vec<f32>>, [f32; OUTPUT_DIM]) {
    let n = x.len();
    let dim = x.first().map(Vec::len).unwrap_or(0);
    let mut w = vec![vec![0.0f32; OUTPUT_DIM]; dim];
    let mut b = [0.0f32; OUTPUT_DIM];

    let scale = 2.0 / n.max(1) as f32;

    for _ in 0..epochs {
        let mut grad_w = vec![vec![0.0f32; OUTPUT_DIM]; dim];
        let mut grad_b = [0.0f32; OUTPUT_DIM];

        for (row, target) in x.iter().zip(y.iter()) {
            let mut pred = [0.0f32; OUTPUT_DIM];
            for o in 0..OUTPUT_DIM {
                pred[o] = b[o];
                for f in 0..dim {
                    pred[o] += row[f] * w[f][o];
                }
            }

            for o in 0..OUTPUT_DIM {
                let err = pred[o] - target[o];
                grad_b[o] += err;
                for f in 0..dim {
                    grad_w[f][o] += row[f] * err;
                }
            }
        }

        for o in 0..OUTPUT_DIM {
            b[o] -= lr * scale * grad_b[o];
        }

        for f in 0..dim {
            for o in 0..OUTPUT_DIM {
                let reg = l2 * w[f][o];
                w[f][o] -= lr * (scale * grad_w[f][o] + reg);
            }
        }
    }

    (w, b)
}

fn mean_std_columns(x: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let dim = x.first().map(Vec::len).unwrap_or(0);
    let n = x.len().max(1) as f32;
    let mut mean = vec![0.0f32; dim];
    for row in x {
        for i in 0..dim {
            mean[i] += row[i] / n;
        }
    }

    let mut var = vec![0.0f32; dim];
    for row in x {
        for i in 0..dim {
            let d = row[i] - mean[i];
            var[i] += d * d / n;
        }
    }
    let std = var.into_iter().map(|v| v.sqrt().max(1e-6)).collect();
    (mean, std)
}

fn normalize_rows(x: &[Vec<f32>], mean: &[f32], std: &[f32]) -> Vec<Vec<f32>> {
    x.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, &v)| (v - mean[i]) / std[i])
                .collect()
        })
        .collect()
}

fn projection_from_weights(w: &[Vec<f32>]) -> LinearProjection {
    LinearProjection {
        duration: col(w, 0),
        predelay_ms: col(w, 1),
        t60: col(w, 2),
        edt: col(w, 3),
        brightness: col(w, 4),
        hf_damping: col(w, 5),
        lf_bloom: col(w, 6),
        spectral_tilt: col(w, 7),
        band_decay_low: col(w, 8),
        band_decay_mid: col(w, 9),
        band_decay_high: col(w, 10),
        early_density: col(w, 11),
        late_density: col(w, 12),
        diffusion: col(w, 13),
        modal_density: col(w, 14),
        tail_noise: col(w, 15),
        grain: col(w, 16),
        width: col(w, 17),
        decorrelation: col(w, 18),
        asymmetry: col(w, 19),
    }
}

fn col(w: &[Vec<f32>], i: usize) -> Vec<f32> {
    w.iter().map(|row| row[i]).collect()
}

fn bias_from_vector(v: &[f32; OUTPUT_DIM]) -> DeltaBias {
    DeltaBias {
        duration: v[0],
        predelay_ms: v[1],
        t60: v[2],
        edt: v[3],
        brightness: v[4],
        hf_damping: v[5],
        lf_bloom: v[6],
        spectral_tilt: v[7],
        band_decay_low: v[8],
        band_decay_mid: v[9],
        band_decay_high: v[10],
        early_density: v[11],
        late_density: v[12],
        diffusion: v[13],
        modal_density: v[14],
        tail_noise: v[15],
        grain: v[16],
        width: v[17],
        decorrelation: v[18],
        asymmetry: v[19],
    }
}
