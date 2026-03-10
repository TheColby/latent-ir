use std::path::PathBuf;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::cli::{AbTestArgs, GenerateArgs};
use crate::core::analysis::AnalysisReport;
use crate::core::util;

use super::generate;

#[derive(Debug, Serialize)]
struct AbTestReport {
    schema_version: String,
    generated_at_utc: DateTime<Utc>,
    prompt: String,
    industrial_text_model: String,
    industrial: VariantArtifacts,
    baseline: VariantArtifacts,
    delta: AnalysisDelta,
}

#[derive(Debug, Serialize)]
struct VariantArtifacts {
    wav: String,
    metadata_json: String,
    analysis_json: String,
    analysis: AnalysisReport,
}

#[derive(Debug, Serialize)]
struct AnalysisDelta {
    t60_s_est: f32,
    predelay_ms_est: f32,
    spectral_centroid_hz: f32,
    early_energy_ratio: f32,
    late_energy_ratio: f32,
    rms: f32,
}

pub fn run(args: AbTestArgs) -> Result<()> {
    std::fs::create_dir_all(&args.output_dir)?;

    let industrial_wav = args.output_dir.join("industrial.wav");
    let industrial_meta = args.output_dir.join("industrial.metadata.json");
    let industrial_analysis = args.output_dir.join("industrial.analysis.json");

    let baseline_wav = args.output_dir.join("baseline.wav");
    let baseline_meta = args.output_dir.join("baseline.metadata.json");
    let baseline_analysis = args.output_dir.join("baseline.analysis.json");

    let industrial_generate = build_generate_args(
        &args,
        industrial_wav.clone(),
        industrial_meta.clone(),
        industrial_analysis.clone(),
        Some(args.industrial_text_model.clone()),
    );
    generate::run(industrial_generate)?;

    let baseline_generate = build_generate_args(
        &args,
        baseline_wav.clone(),
        baseline_meta.clone(),
        baseline_analysis.clone(),
        None,
    );
    generate::run(baseline_generate)?;

    let industrial_report: AnalysisReport =
        serde_json::from_str(&std::fs::read_to_string(&industrial_analysis)?)?;
    let baseline_report: AnalysisReport =
        serde_json::from_str(&std::fs::read_to_string(&baseline_analysis)?)?;

    let delta = AnalysisDelta {
        t60_s_est: industrial_report.t60_s_est.unwrap_or(0.0)
            - baseline_report.t60_s_est.unwrap_or(0.0),
        predelay_ms_est: industrial_report.predelay_ms_est - baseline_report.predelay_ms_est,
        spectral_centroid_hz: industrial_report.spectral_centroid_hz
            - baseline_report.spectral_centroid_hz,
        early_energy_ratio: industrial_report.early_energy_ratio
            - baseline_report.early_energy_ratio,
        late_energy_ratio: industrial_report.late_energy_ratio - baseline_report.late_energy_ratio,
        rms: industrial_report.rms - baseline_report.rms,
    };

    let report = AbTestReport {
        schema_version: "latent-ir.ab-test.v1".to_string(),
        generated_at_utc: Utc::now(),
        prompt: args.prompt,
        industrial_text_model: args.industrial_text_model.display().to_string(),
        industrial: VariantArtifacts {
            wav: industrial_wav.display().to_string(),
            metadata_json: industrial_meta.display().to_string(),
            analysis_json: industrial_analysis.display().to_string(),
            analysis: industrial_report,
        },
        baseline: VariantArtifacts {
            wav: baseline_wav.display().to_string(),
            metadata_json: baseline_meta.display().to_string(),
            analysis_json: baseline_analysis.display().to_string(),
            analysis: baseline_report,
        },
        delta,
    };

    let report_path = args.output_dir.join("ab_test_report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

    if args.markdown {
        let md = render_markdown_scorecard(&report);
        let md_path = args.output_dir.join("ab_test_report.md");
        std::fs::write(&md_path, md)?;
        println!(
            "{}",
            util::console::info("wrote A/B markdown", md_path.display().to_string())
        );
    }

    println!(
        "{}",
        util::console::info("wrote A/B report", report_path.display().to_string())
    );
    println!(
        "{}",
        util::console::info("industrial wav", report.industrial.wav.clone())
    );
    println!(
        "{}",
        util::console::info("baseline wav", report.baseline.wav.clone())
    );
    println!(
        "{}",
        util::console::metric("delta t60_s_est", format!("{:.3}", report.delta.t60_s_est))
    );
    Ok(())
}

fn render_markdown_scorecard(report: &AbTestReport) -> String {
    format!(
        "# latent-ir A/B Scorecard\n\n\
Generated: {}\n\n\
Prompt: `{}`\n\n\
Industrial model: `{}`\n\n\
## Key Deltas (Industrial - Baseline)\n\n\
| Metric | Delta |\n|---|---:|\n\
| T60 (s) | {:.3} |\n\
| Predelay (ms) | {:.3} |\n\
| Spectral Centroid (Hz) | {:.3} |\n\
| Early Energy Ratio | {:.4} |\n\
| Late Energy Ratio | {:.4} |\n\
| RMS | {:.4} |\n\n\
## Artifact Paths\n\n\
- Industrial WAV: `{}`\n\
- Baseline WAV: `{}`\n\
- Industrial Analysis JSON: `{}`\n\
- Baseline Analysis JSON: `{}`\n",
        report.generated_at_utc,
        report.prompt,
        report.industrial_text_model,
        report.delta.t60_s_est,
        report.delta.predelay_ms_est,
        report.delta.spectral_centroid_hz,
        report.delta.early_energy_ratio,
        report.delta.late_energy_ratio,
        report.delta.rms,
        report.industrial.wav,
        report.baseline.wav,
        report.industrial.analysis_json,
        report.baseline.analysis_json
    )
}

fn build_generate_args(
    args: &AbTestArgs,
    output: PathBuf,
    metadata_out: PathBuf,
    json_analysis_out: PathBuf,
    text_encoder_model: Option<PathBuf>,
) -> GenerateArgs {
    GenerateArgs {
        prompt: Some(args.prompt.clone()),
        text_encoder_model,
        text_encoder_onnx: None,
        text_encoder_onnx_input_dim: 256,
        reference_audio: None,
        audio_encoder_model: None,
        audio_encoder_onnx: None,
        preset: args.preset.clone(),
        output,
        metadata_out: Some(metadata_out),
        json_analysis_out: Some(json_analysis_out),
        sample_rate: args.sample_rate,
        seed: args.seed,
        duration: args.duration,
        t60: args.t60,
        predelay_ms: args.predelay_ms,
        edt: args.edt,
        brightness: None,
        diffusion: None,
        early_density: None,
        late_density: None,
        width: None,
        decorrelation: None,
        macro_size: args.macro_size,
        macro_distance: args.macro_distance,
        macro_material: args.macro_material,
        macro_clarity: args.macro_clarity,
        macro_trajectory: args.macro_trajectory.clone(),
        channels: args.channels,
    }
}
