use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::analysis::{AnalysisReport, IrAnalyzer};
use crate::core::conditioning::{
    AudioEncoder, LearnedAudioEncoder, LearnedTextEncoder, TextEncoder,
};
use crate::core::descriptors::DescriptorSet;
use crate::core::generator::{IrGenerator, ProceduralIrGenerator};
use crate::core::training::{AudioTrainSample, TextTrainSample};
use crate::core::util;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineReport {
    pub schema_version: String,
    pub command: String,
    pub generated_at_utc: DateTime<Utc>,
    pub dataset_path: String,
    pub model_path: String,
    pub sample_count: usize,
    pub descriptor_metrics: DescriptorMetrics,
    pub analysis_metrics: AnalysisMetrics,
    #[serde(default)]
    pub uncertainty_metrics: UncertaintyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptorMetrics {
    pub mae: f32,
    pub rmse: f32,
    pub per_field_mae: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    pub mae: f32,
    pub rmse: f32,
    pub per_metric_mae: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UncertaintyMetrics {
    pub mean_confidence: f32,
    pub mean_uncertainty: f32,
    pub per_field_confidence: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCheckResult {
    pub passed: bool,
    pub max_regression: f32,
    pub regressions: Vec<String>,
}

pub fn evaluate_text(
    dataset_path: &Path,
    model_path: &Path,
    sample_rate: u32,
    seed: u64,
) -> Result<BaselineReport> {
    let samples = load_text_dataset(dataset_path)?;
    let model = LearnedTextEncoder::from_json_file(model_path)?;
    let predicted: Vec<DescriptorSet> = samples
        .iter()
        .map(|s| {
            let mut d = DescriptorSet::default();
            let delta = model.infer_delta_from_prompt(&s.prompt)?;
            delta.apply_to(&mut d, 1.0);
            Ok(d)
        })
        .collect::<Result<_>>()?;

    let targets: Vec<DescriptorSet> = samples.iter().map(|s| s.descriptor.clone()).collect();
    let descriptor_metrics = descriptor_metrics(&predicted, &targets);
    let analysis_metrics = analysis_metrics(&predicted, &targets, sample_rate, seed)?;
    let uncertainty_metrics = uncertainty_metrics(&predicted, &targets);

    Ok(BaselineReport {
        schema_version: "latent-ir.eval.baseline.v1".to_string(),
        command: "eval text".to_string(),
        generated_at_utc: Utc::now(),
        dataset_path: dataset_path.display().to_string(),
        model_path: model_path.display().to_string(),
        sample_count: samples.len(),
        descriptor_metrics,
        analysis_metrics,
        uncertainty_metrics,
    })
}

pub fn evaluate_audio(
    dataset_path: &Path,
    model_path: &Path,
    sample_rate: u32,
    seed: u64,
) -> Result<BaselineReport> {
    let samples = load_audio_dataset(dataset_path)?;
    let model = LearnedAudioEncoder::from_json_file(model_path)?;
    let root = dataset_path.parent().unwrap_or_else(|| Path::new("."));

    let predicted: Vec<DescriptorSet> = samples
        .iter()
        .map(|s| {
            let audio_path = resolve_audio_path(root, &s.audio_path);
            let wav = util::audio::read_wav_f32(&audio_path)
                .with_context(|| format!("failed to read {}", audio_path.display()))?;
            let mut d = DescriptorSet::default();
            let delta = model.infer_delta_from_audio(&wav.channels, wav.sample_rate)?;
            delta.apply_to(&mut d, 1.0);
            Ok(d)
        })
        .collect::<Result<_>>()?;

    let targets: Vec<DescriptorSet> = samples.iter().map(|s| s.descriptor.clone()).collect();
    let descriptor_metrics = descriptor_metrics(&predicted, &targets);
    let analysis_metrics = analysis_metrics(&predicted, &targets, sample_rate, seed)?;
    let uncertainty_metrics = uncertainty_metrics(&predicted, &targets);

    Ok(BaselineReport {
        schema_version: "latent-ir.eval.baseline.v1".to_string(),
        command: "eval audio".to_string(),
        generated_at_utc: Utc::now(),
        dataset_path: dataset_path.display().to_string(),
        model_path: model_path.display().to_string(),
        sample_count: samples.len(),
        descriptor_metrics,
        analysis_metrics,
        uncertainty_metrics,
    })
}

pub fn check_eval(
    report: &BaselineReport,
    baseline: &BaselineReport,
    max_regression: f32,
) -> EvalCheckResult {
    let mut regressions = Vec::new();
    check_metric(
        "descriptor_metrics.mae",
        report.descriptor_metrics.mae,
        baseline.descriptor_metrics.mae,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "descriptor_metrics.rmse",
        report.descriptor_metrics.rmse,
        baseline.descriptor_metrics.rmse,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "analysis_metrics.mae",
        report.analysis_metrics.mae,
        baseline.analysis_metrics.mae,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "analysis_metrics.rmse",
        report.analysis_metrics.rmse,
        baseline.analysis_metrics.rmse,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "uncertainty_metrics.mean_uncertainty",
        report.uncertainty_metrics.mean_uncertainty,
        baseline.uncertainty_metrics.mean_uncertainty,
        max_regression,
        &mut regressions,
    );

    EvalCheckResult {
        passed: regressions.is_empty(),
        max_regression,
        regressions,
    }
}

fn load_text_dataset(path: &Path) -> Result<Vec<TextTrainSample>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read dataset {}", path.display()))?;
    let samples: Vec<TextTrainSample> =
        serde_json::from_str(&text).with_context(|| "failed to parse text eval dataset JSON")?;
    Ok(samples)
}

fn load_audio_dataset(path: &Path) -> Result<Vec<AudioTrainSample>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read dataset {}", path.display()))?;
    let samples: Vec<AudioTrainSample> =
        serde_json::from_str(&text).with_context(|| "failed to parse audio eval dataset JSON")?;
    Ok(samples)
}

fn resolve_audio_path(root: &Path, p: &str) -> PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

fn descriptor_metrics(pred: &[DescriptorSet], tgt: &[DescriptorSet]) -> DescriptorMetrics {
    let names = descriptor_field_names();
    let mut sums = vec![0.0f32; names.len()];
    let mut sums_sq = vec![0.0f32; names.len()];
    let mut n_total = 0usize;

    for (p, t) in pred.iter().zip(tgt.iter()) {
        let pv = descriptor_to_vec(p);
        let tv = descriptor_to_vec(t);
        for i in 0..pv.len() {
            let e = (pv[i] - tv[i]).abs();
            sums[i] += e;
            sums_sq[i] += e * e;
            n_total += 1;
        }
    }

    let n = pred.len().max(1) as f32;
    let mut per = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        per.insert((*name).to_string(), sums[i] / n);
    }

    let mae = sums.iter().sum::<f32>() / n_total.max(1) as f32;
    let rmse = (sums_sq.iter().sum::<f32>() / n_total.max(1) as f32).sqrt();

    DescriptorMetrics {
        mae,
        rmse,
        per_field_mae: per,
    }
}

fn analysis_metrics(
    pred: &[DescriptorSet],
    tgt: &[DescriptorSet],
    sample_rate: u32,
    seed: u64,
) -> Result<AnalysisMetrics> {
    let generator = ProceduralIrGenerator::new(sample_rate);
    let analyzer = IrAnalyzer;

    let names = analysis_metric_names();
    let mut sums = vec![0.0f32; names.len()];
    let mut sums_sq = vec![0.0f32; names.len()];
    let mut n_total = 0usize;

    for (i, (p, t)) in pred.iter().zip(tgt.iter()).enumerate() {
        let a = synth_analyze(
            &generator,
            &analyzer,
            p,
            sample_rate,
            seed.wrapping_add(i as u64),
        )?;
        let b = synth_analyze(
            &generator,
            &analyzer,
            t,
            sample_rate,
            seed.wrapping_add(i as u64).wrapping_add(9_999),
        )?;
        let av = analysis_to_vec(&a);
        let bv = analysis_to_vec(&b);
        for k in 0..av.len() {
            let e = (av[k] - bv[k]).abs();
            sums[k] += e;
            sums_sq[k] += e * e;
            n_total += 1;
        }
    }

    let n = pred.len().max(1) as f32;
    let mut per = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        per.insert((*name).to_string(), sums[i] / n);
    }

    let mae = sums.iter().sum::<f32>() / n_total.max(1) as f32;
    let rmse = (sums_sq.iter().sum::<f32>() / n_total.max(1) as f32).sqrt();

    Ok(AnalysisMetrics {
        mae,
        rmse,
        per_metric_mae: per,
    })
}

fn synth_analyze(
    generator: &ProceduralIrGenerator,
    analyzer: &IrAnalyzer,
    descriptor: &DescriptorSet,
    sample_rate: u32,
    seed: u64,
) -> Result<AnalysisReport> {
    let ir = generator.generate(descriptor, seed)?;
    Ok(analyzer.analyze(&ir.channels, sample_rate))
}

fn descriptor_to_vec(d: &DescriptorSet) -> [f32; 20] {
    [
        d.time.duration,
        d.time.predelay_ms,
        d.time.t60,
        d.time.edt,
        d.spectral.brightness,
        d.spectral.hf_damping,
        d.spectral.lf_bloom,
        d.spectral.spectral_tilt,
        d.spectral.band_decay_low,
        d.spectral.band_decay_mid,
        d.spectral.band_decay_high,
        d.structural.early_density,
        d.structural.late_density,
        d.structural.diffusion,
        d.structural.modal_density,
        d.structural.tail_noise,
        d.structural.grain,
        d.spatial.width,
        d.spatial.decorrelation,
        d.spatial.asymmetry,
    ]
}

fn descriptor_field_names() -> [&'static str; 20] {
    [
        "duration",
        "predelay_ms",
        "t60",
        "edt",
        "brightness",
        "hf_damping",
        "lf_bloom",
        "spectral_tilt",
        "band_decay_low",
        "band_decay_mid",
        "band_decay_high",
        "early_density",
        "late_density",
        "diffusion",
        "modal_density",
        "tail_noise",
        "grain",
        "width",
        "decorrelation",
        "asymmetry",
    ]
}

fn analysis_to_vec(r: &AnalysisReport) -> [f32; 11] {
    [
        r.duration_s,
        r.peak,
        r.rms,
        r.predelay_ms_est,
        r.edt_s_est.unwrap_or(0.0),
        r.t20_s_est.unwrap_or(0.0),
        r.t30_s_est.unwrap_or(0.0),
        r.t60_s_est.unwrap_or(0.0),
        r.spectral_centroid_hz,
        r.early_energy_ratio,
        r.late_energy_ratio,
    ]
}

fn analysis_metric_names() -> [&'static str; 11] {
    [
        "duration_s",
        "peak",
        "rms",
        "predelay_ms_est",
        "edt_s_est",
        "t20_s_est",
        "t30_s_est",
        "t60_s_est",
        "spectral_centroid_hz",
        "early_energy_ratio",
        "late_energy_ratio",
    ]
}

fn uncertainty_metrics(pred: &[DescriptorSet], tgt: &[DescriptorSet]) -> UncertaintyMetrics {
    let names = descriptor_field_names();
    let mut sums = vec![0.0f32; names.len()];
    let mut n = 0usize;

    for (p, t) in pred.iter().zip(tgt.iter()) {
        let pv = descriptor_to_vec(p);
        let tv = descriptor_to_vec(t);
        for i in 0..pv.len() {
            let abs_err = (pv[i] - tv[i]).abs();
            let conf = (1.0 / (1.0 + abs_err / (tv[i].abs() + 0.05))).clamp(0.0, 1.0);
            sums[i] += conf;
        }
        n += 1;
    }

    let denom = n.max(1) as f32;
    let mut per_field_confidence = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        per_field_confidence.insert((*name).to_string(), sums[i] / denom);
    }
    let mean_confidence = per_field_confidence.values().copied().sum::<f32>()
        / per_field_confidence.len().max(1) as f32;
    UncertaintyMetrics {
        mean_confidence,
        mean_uncertainty: (1.0 - mean_confidence).clamp(0.0, 1.0),
        per_field_confidence,
    }
}

fn check_metric(
    name: &str,
    current: f32,
    baseline: f32,
    max_regression: f32,
    regressions: &mut Vec<String>,
) {
    if baseline.abs() <= 1e-9 {
        return;
    }
    let rel = (current - baseline) / baseline.abs();
    if rel > max_regression {
        regressions.push(format!(
            "{name}: current={current:.6} baseline={baseline:.6} regression={:.2}%",
            rel * 100.0
        ));
    }
}
