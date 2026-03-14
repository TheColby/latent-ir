use std::path::{Path, PathBuf};

use latent_ir::core::analysis::IrAnalyzer;
use latent_ir::core::descriptors::{ChannelFormat, DescriptorSet};
use latent_ir::core::generator::{IrGenerator, ProceduralIrGenerator};
use latent_ir::core::semantics::SemanticResolver;
use latent_ir::core::spatial::{build_channel_map, load_custom_layout_file};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SpatialCorpus {
    schema_version: String,
    samples: Vec<SpatialSample>,
}

#[derive(Debug, Deserialize)]
struct SpatialSample {
    id: String,
    prompt: Option<String>,
    channels: String,
    layout_json: Option<String>,
    seed: u64,
    sample_rate: u32,
    duration: f32,
    #[serde(default)]
    t60: Option<f32>,
    expected: ExpectedMetrics,
}

#[derive(Debug, Deserialize)]
struct ExpectedMetrics {
    channels: usize,
    #[serde(default)]
    inter_channel_correlation_mean_abs: Option<RangeEnvelope>,
    #[serde(default)]
    front_energy_ratio: Option<RangeEnvelope>,
    #[serde(default)]
    rear_energy_ratio: Option<RangeEnvelope>,
    #[serde(default)]
    height_energy_ratio: Option<RangeEnvelope>,
    #[serde(default)]
    lfe_energy_ratio: Option<RangeEnvelope>,
}

#[derive(Debug, Deserialize)]
struct RangeEnvelope {
    min: f32,
    max: f32,
}

#[test]
fn spatial_corpus_metrics_stay_within_envelopes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpus_path = root.join("ci/datasets/spatial_corpus_ci.json");
    let corpus: SpatialCorpus =
        serde_json::from_str(&std::fs::read_to_string(&corpus_path).expect("corpus file"))
            .expect("valid corpus JSON");

    assert_eq!(corpus.schema_version, "latent-ir.spatial-corpus.v1");
    assert!(!corpus.samples.is_empty());

    let resolver = SemanticResolver::default();
    let analyzer = IrAnalyzer;

    for sample in &corpus.samples {
        let mut d = DescriptorSet::default();
        if let Some(prompt) = sample.prompt.as_deref() {
            resolver.apply_prompt(prompt, &mut d);
        }
        d.time.duration = sample.duration;
        if let Some(t60) = sample.t60 {
            d.time.t60 = t60;
        }
        d.spatial.channel_format =
            parse_channel_format(&sample.channels).expect("supported channel format in corpus");

        if let Some(layout_rel) = sample.layout_json.as_deref() {
            let layout_path = resolve_path(&root, layout_rel);
            let layout = load_custom_layout_file(&layout_path).expect("custom layout must parse");
            d.spatial.set_custom_layout(layout);
        }
        d.clamp();

        let generator = ProceduralIrGenerator::new(sample.sample_rate);
        let ir = generator
            .generate(&d, sample.seed)
            .expect("corpus sample generation");
        assert_eq!(
            ir.channels.len(),
            sample.expected.channels,
            "sample '{}' channel count mismatch",
            sample.id
        );

        let channel_map = build_channel_map(&d.spatial);
        let report =
            analyzer.analyze_with_channel_map(&ir.channels, sample.sample_rate, Some(&channel_map));
        assert_eq!(
            report.channels, sample.expected.channels,
            "sample '{}' analysis channel count mismatch",
            sample.id
        );

        check_envelope(
            sample,
            "inter_channel_correlation_mean_abs",
            report.inter_channel_correlation_mean_abs,
            sample.expected.inter_channel_correlation_mean_abs.as_ref(),
        );
        check_envelope(
            sample,
            "front_energy_ratio",
            report.front_energy_ratio,
            sample.expected.front_energy_ratio.as_ref(),
        );
        check_envelope(
            sample,
            "rear_energy_ratio",
            report.rear_energy_ratio,
            sample.expected.rear_energy_ratio.as_ref(),
        );
        check_envelope(
            sample,
            "height_energy_ratio",
            report.height_energy_ratio,
            sample.expected.height_energy_ratio.as_ref(),
        );
        check_envelope(
            sample,
            "lfe_energy_ratio",
            report.lfe_energy_ratio,
            sample.expected.lfe_energy_ratio.as_ref(),
        );
    }
}

fn parse_channel_format(s: &str) -> Option<ChannelFormat> {
    match s.trim().to_ascii_lowercase().as_str() {
        "mono" => Some(ChannelFormat::Mono),
        "stereo" => Some(ChannelFormat::Stereo),
        "foa" | "foa_ambix" => Some(ChannelFormat::FoaAmbix),
        "5.1" | "surround_5_1" => Some(ChannelFormat::Surround5_1),
        "7.1" | "surround_7_1" => Some(ChannelFormat::Surround7_1),
        "7.1.4" | "atmos_7_1_4" => Some(ChannelFormat::Atmos7_1_4),
        "7.2.4" | "atmos_7_2_4" => Some(ChannelFormat::Atmos7_2_4),
        "custom" => Some(ChannelFormat::Custom),
        _ => None,
    }
}

fn resolve_path(root: &Path, rel_or_abs: &str) -> PathBuf {
    let p = Path::new(rel_or_abs);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

fn check_envelope(
    sample: &SpatialSample,
    name: &str,
    value: Option<f32>,
    env: Option<&RangeEnvelope>,
) {
    if let Some(env) = env {
        let v = value.unwrap_or_else(|| panic!("sample '{}' missing {}", sample.id, name));
        assert!(
            v >= env.min && v <= env.max,
            "sample '{}' {} out of range [{:.4}, {:.4}] -> {:.6}",
            sample.id,
            name,
            env.min,
            env.max,
            v
        );
    }
}
