use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::analysis::{AnalysisReport, IrAnalyzer};
use crate::core::conditioning::{
    AudioEncoder, LearnedAudioEncoder, LearnedTextEncoder, TextEncoder,
};
use crate::core::descriptors::DescriptorSet;
use crate::core::generator::{IrGenerator, ProceduralIrGenerator};
use crate::core::semantics::SemanticResolver;
use crate::core::util;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSample {
    pub id: String,
    pub prompt: Option<String>,
    pub reference_audio: Option<String>,
    pub target_ir: Option<String>,
    pub target_descriptor: Option<DescriptorSet>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    pub schema_version: String,
    pub samples: Vec<BenchmarkSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: String,
    pub generated_at_utc: DateTime<Utc>,
    pub dataset_path: String,
    pub sample_count: usize,
    pub repeats: usize,
    pub objective: ObjectiveMetrics,
    pub speed: SpeedMetrics,
    pub stability: StabilityMetrics,
    pub perceptual_proxy: PerceptualProxyMetrics,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveMetrics {
    pub descriptor_mae: f32,
    pub descriptor_rmse: f32,
    pub analysis_mae: f32,
    pub analysis_rmse: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedMetrics {
    pub encode_ms_avg: f32,
    pub generate_ms_avg: f32,
    pub analyze_ms_avg: f32,
    pub total_ms_avg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub descriptor_std_avg: f32,
    pub analysis_std_avg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualProxyMetrics {
    pub proxy_mae: f32,
    pub per_proxy_mae: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub objective_score: f32,
    pub speed_score: f32,
    pub stability_score: f32,
    pub perceptual_score: f32,
    pub total_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCheckResult {
    pub passed: bool,
    pub max_regression: f32,
    pub regressions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkRunConfig {
    pub dataset_path: PathBuf,
    pub sample_rate: u32,
    pub seed: u64,
    pub repeats: usize,
    pub text_model: Option<PathBuf>,
    pub audio_model: Option<PathBuf>,
}

pub fn run_benchmark(cfg: &BenchmarkRunConfig) -> Result<BenchmarkReport> {
    let dataset = load_dataset(&cfg.dataset_path)?;
    if dataset.samples.is_empty() {
        return Err(anyhow!("benchmark dataset has no samples"));
    }

    let text_model = if let Some(path) = cfg.text_model.as_deref() {
        Some(LearnedTextEncoder::from_json_file(path)?)
    } else {
        None
    };
    let audio_model = if let Some(path) = cfg.audio_model.as_deref() {
        Some(LearnedAudioEncoder::from_json_file(path)?)
    } else {
        None
    };

    let generator = ProceduralIrGenerator::new(cfg.sample_rate);
    let analyzer = IrAnalyzer;
    let dataset_root = cfg.dataset_path.parent().unwrap_or_else(|| Path::new("."));

    let mut descriptor_abs = Vec::new();
    let mut descriptor_sq = Vec::new();
    let mut analysis_abs = Vec::new();
    let mut analysis_sq = Vec::new();

    let mut encode_ms = 0.0f64;
    let mut generate_ms = 0.0f64;
    let mut analyze_ms = 0.0f64;
    let mut total_ms = 0.0f64;

    let mut descriptor_stability = Vec::new();
    let mut analysis_stability = Vec::new();

    let mut proxy_abs = vec![0.0f32; 4];
    let mut proxy_n = 0usize;

    for (i, sample) in dataset.samples.iter().enumerate() {
        let t_total = Instant::now();

        let t_enc = Instant::now();
        let predicted = infer_descriptor(sample, &text_model, &audio_model, dataset_root)?;
        encode_ms += t_enc.elapsed().as_secs_f64() * 1000.0;

        let t_gen = Instant::now();
        let generated = generator.generate(&predicted, cfg.seed.wrapping_add(i as u64))?;
        generate_ms += t_gen.elapsed().as_secs_f64() * 1000.0;

        let t_an = Instant::now();
        let pred_analysis = analyzer.analyze(&generated.channels, cfg.sample_rate);
        analyze_ms += t_an.elapsed().as_secs_f64() * 1000.0;

        let target_desc = sample
            .target_descriptor
            .clone()
            .unwrap_or_else(DescriptorSet::default);

        let target_analysis = if let Some(ir_path) = sample.target_ir.as_deref() {
            let wav = util::audio::read_wav_f32(resolve_path(dataset_root, Path::new(ir_path)))?;
            analyzer.analyze(&wav.channels, wav.sample_rate)
        } else {
            let synth = generator.generate(
                &target_desc,
                cfg.seed.wrapping_add(i as u64).wrapping_add(100_000),
            )?;
            analyzer.analyze(&synth.channels, cfg.sample_rate)
        };

        accumulate_errors(
            &descriptor_to_vec(&predicted),
            &descriptor_to_vec(&target_desc),
            &mut descriptor_abs,
            &mut descriptor_sq,
        );
        accumulate_errors(
            &analysis_to_vec(&pred_analysis),
            &analysis_to_vec(&target_analysis),
            &mut analysis_abs,
            &mut analysis_sq,
        );

        let p_pred = analysis_proxies(&pred_analysis, cfg.sample_rate);
        let p_tgt = analysis_proxies(&target_analysis, cfg.sample_rate);
        for k in 0..proxy_abs.len() {
            proxy_abs[k] += (p_pred[k] - p_tgt[k]).abs();
        }
        proxy_n += 1;

        let (desc_std, an_std) = repeat_stability(
            sample,
            &text_model,
            &audio_model,
            dataset_root,
            &generator,
            &analyzer,
            cfg,
            i,
        )?;
        descriptor_stability.push(desc_std);
        analysis_stability.push(an_std);

        total_ms += t_total.elapsed().as_secs_f64() * 1000.0;
    }

    let n = dataset.samples.len() as f32;
    let descriptor_mae =
        descriptor_abs.iter().sum::<f32>() / (descriptor_abs.len().max(1) as f32 * n);
    let descriptor_rmse =
        (descriptor_sq.iter().sum::<f32>() / (descriptor_sq.len().max(1) as f32 * n)).sqrt();
    let analysis_mae = analysis_abs.iter().sum::<f32>() / (analysis_abs.len().max(1) as f32 * n);
    let analysis_rmse =
        (analysis_sq.iter().sum::<f32>() / (analysis_sq.len().max(1) as f32 * n)).sqrt();

    let objective = ObjectiveMetrics {
        descriptor_mae,
        descriptor_rmse,
        analysis_mae,
        analysis_rmse,
    };

    let speed = SpeedMetrics {
        encode_ms_avg: (encode_ms as f32) / n,
        generate_ms_avg: (generate_ms as f32) / n,
        analyze_ms_avg: (analyze_ms as f32) / n,
        total_ms_avg: (total_ms as f32) / n,
    };

    let stability = StabilityMetrics {
        descriptor_std_avg: mean(&descriptor_stability),
        analysis_std_avg: mean(&analysis_stability),
    };

    let mut per_proxy_mae = BTreeMap::new();
    let names = ["clarity", "brightness", "spaciousness", "distance"];
    for (i, name) in names.iter().enumerate() {
        per_proxy_mae.insert((*name).to_string(), proxy_abs[i] / proxy_n.max(1) as f32);
    }
    let perceptual_proxy = PerceptualProxyMetrics {
        proxy_mae: proxy_abs.iter().sum::<f32>()
            / (proxy_abs.len().max(1) as f32 * proxy_n.max(1) as f32),
        per_proxy_mae,
    };

    let summary = BenchmarkSummary {
        objective_score: objective.descriptor_mae + objective.analysis_mae,
        speed_score: speed.total_ms_avg / 1000.0,
        stability_score: stability.descriptor_std_avg + stability.analysis_std_avg,
        perceptual_score: perceptual_proxy.proxy_mae,
        total_score: objective.descriptor_mae
            + objective.analysis_mae
            + speed.total_ms_avg / 1000.0
            + stability.descriptor_std_avg
            + stability.analysis_std_avg
            + perceptual_proxy.proxy_mae,
    };

    Ok(BenchmarkReport {
        schema_version: "latent-ir.benchmark.v1".to_string(),
        generated_at_utc: Utc::now(),
        dataset_path: cfg.dataset_path.display().to_string(),
        sample_count: dataset.samples.len(),
        repeats: cfg.repeats,
        objective,
        speed,
        stability,
        perceptual_proxy,
        summary,
    })
}

pub fn check_benchmark(
    report: &BenchmarkReport,
    baseline: &BenchmarkReport,
    max_regression: f32,
) -> BenchmarkCheckResult {
    let mut regressions = Vec::new();
    check_metric(
        "summary.total_score",
        report.summary.total_score,
        baseline.summary.total_score,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "objective.descriptor_mae",
        report.objective.descriptor_mae,
        baseline.objective.descriptor_mae,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "objective.analysis_mae",
        report.objective.analysis_mae,
        baseline.objective.analysis_mae,
        max_regression,
        &mut regressions,
    );
    check_metric(
        "perceptual_proxy.proxy_mae",
        report.perceptual_proxy.proxy_mae,
        baseline.perceptual_proxy.proxy_mae,
        max_regression,
        &mut regressions,
    );

    BenchmarkCheckResult {
        passed: regressions.is_empty(),
        max_regression,
        regressions,
    }
}

fn infer_descriptor(
    sample: &BenchmarkSample,
    text_model: &Option<LearnedTextEncoder>,
    audio_model: &Option<LearnedAudioEncoder>,
    root: &Path,
) -> Result<DescriptorSet> {
    let mut d = DescriptorSet::default();

    if let (Some(model), Some(prompt)) = (text_model, sample.prompt.as_deref()) {
        let delta = model.infer_delta_from_prompt(prompt)?;
        delta.apply_to(&mut d, 1.0);
    } else if let Some(prompt) = sample.prompt.as_deref() {
        SemanticResolver::default().apply_prompt(prompt, &mut d);
    }

    if let (Some(model), Some(audio_path)) = (audio_model, sample.reference_audio.as_deref()) {
        let wav = util::audio::read_wav_f32(resolve_path(root, Path::new(audio_path)))?;
        let delta = model.infer_delta_from_audio(&wav.channels, wav.sample_rate)?;
        delta.apply_to(&mut d, 1.0);
    }

    d.clamp();
    Ok(d)
}

fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn load_dataset(path: &Path) -> Result<BenchmarkDataset> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read benchmark dataset {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| "failed to parse benchmark dataset JSON")
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

fn accumulate_errors(pred: &[f32], tgt: &[f32], abs_out: &mut Vec<f32>, sq_out: &mut Vec<f32>) {
    if abs_out.is_empty() {
        abs_out.resize(pred.len(), 0.0);
        sq_out.resize(pred.len(), 0.0);
    }
    for i in 0..pred.len() {
        let e = (pred[i] - tgt[i]).abs();
        abs_out[i] += e;
        sq_out[i] += e * e;
    }
}

fn analysis_proxies(a: &AnalysisReport, sample_rate: u32) -> [f32; 4] {
    let clarity = a.early_energy_ratio / (a.late_energy_ratio + 1e-6);
    let brightness = (a.spectral_centroid_hz / (sample_rate as f32 / 2.0)).clamp(0.0, 1.0);
    let spaciousness = 1.0 - a.stereo_correlation.unwrap_or(1.0).abs();
    let distance = (a.predelay_ms_est / 200.0).clamp(0.0, 1.0)
        + (a.t60_s_est.unwrap_or(0.0) / 20.0).clamp(0.0, 1.0);
    [clarity, brightness, spaciousness, distance]
}

fn repeat_stability(
    sample: &BenchmarkSample,
    text_model: &Option<LearnedTextEncoder>,
    audio_model: &Option<LearnedAudioEncoder>,
    dataset_root: &Path,
    generator: &ProceduralIrGenerator,
    analyzer: &IrAnalyzer,
    cfg: &BenchmarkRunConfig,
    sample_idx: usize,
) -> Result<(f32, f32)> {
    let repeats = cfg.repeats.max(1);
    if repeats == 1 {
        return Ok((0.0, 0.0));
    }

    let mut desc_rows = Vec::with_capacity(repeats);
    let mut an_rows = Vec::with_capacity(repeats);
    for r in 0..repeats {
        let d = infer_descriptor(sample, text_model, audio_model, dataset_root)?;
        let ir = generator.generate(
            &d,
            cfg.seed
                .wrapping_add(sample_idx as u64)
                .wrapping_add(r as u64),
        )?;
        let an = analyzer.analyze(&ir.channels, cfg.sample_rate);
        desc_rows.push(descriptor_to_vec(&d).to_vec());
        an_rows.push(analysis_to_vec(&an).to_vec());
    }

    Ok((mean_std_rows(&desc_rows), mean_std_rows(&an_rows)))
}

fn mean_std_rows(rows: &[Vec<f32>]) -> f32 {
    if rows.is_empty() {
        return 0.0;
    }
    let dim = rows[0].len();
    let n = rows.len() as f32;
    let mut mean = vec![0.0f32; dim];
    for row in rows {
        for i in 0..dim {
            mean[i] += row[i] / n;
        }
    }

    let mut var = vec![0.0f32; dim];
    for row in rows {
        for i in 0..dim {
            let d = row[i] - mean[i];
            var[i] += d * d / n;
        }
    }
    var.into_iter().map(|v| v.sqrt()).sum::<f32>() / dim.max(1) as f32
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn check_metric(
    name: &str,
    candidate: f32,
    baseline: f32,
    max_regression: f32,
    out: &mut Vec<String>,
) {
    if baseline <= 1e-9 {
        if candidate > baseline + max_regression {
            out.push(format!(
                "{name} regressed: candidate={candidate:.6} baseline={baseline:.6}"
            ));
        }
        return;
    }
    let rel = (candidate - baseline) / baseline;
    if rel > max_regression {
        out.push(format!(
            "{name} regressed by {:.2}% (candidate={:.6} baseline={:.6})",
            rel * 100.0,
            candidate,
            baseline
        ));
    }
}
