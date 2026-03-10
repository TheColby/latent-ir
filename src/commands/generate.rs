use anyhow::{Context, Result};
use chrono::Utc;

use crate::cli::{ChannelFormatArg, GenerateArgs};
use crate::core::analysis::IrAnalyzer;
use crate::core::descriptors::{ChannelFormat, DescriptorSet};
use crate::core::generator::{IrGenerator, ProceduralIrGenerator};
use crate::core::presets;
use crate::core::semantics::SemanticResolver;
use crate::core::util::{self, metadata::GenerationMetadata};

pub fn run(args: GenerateArgs) -> Result<()> {
    let mut descriptor = DescriptorSet::default();

    if let Some(name) = args.preset.as_deref() {
        descriptor =
            presets::resolve_preset(name).with_context(|| format!("unknown preset '{name}'"))?;
    }

    if let Some(prompt) = args.prompt.as_deref() {
        SemanticResolver::default().apply_prompt(prompt, &mut descriptor);
    }

    descriptor.apply_overrides(args.duration, args.t60, args.predelay_ms, args.edt);
    descriptor.apply_spectral_overrides(args.brightness, None, None, None);
    descriptor.apply_structure_overrides(args.early_density, args.late_density, args.diffusion);
    descriptor.apply_spatial_overrides(
        Some(match args.channels {
            ChannelFormatArg::Mono => ChannelFormat::Mono,
            ChannelFormatArg::Stereo => ChannelFormat::Stereo,
        }),
        args.width,
        args.decorrelation,
        None,
    );
    descriptor.clamp();

    let generator = ProceduralIrGenerator::new(args.sample_rate);
    let generated = generator.generate(&descriptor, args.seed)?;

    util::audio::write_wav_f32(&args.output, args.sample_rate, &generated.channels)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    let analysis = IrAnalyzer::default().analyze(&generated.channels, args.sample_rate);

    let metadata = GenerationMetadata {
        schema_version: "latent-ir.generation.v1".to_string(),
        project: "latent-ir".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        command: "generate".to_string(),
        seed: args.seed,
        prompt: args.prompt,
        preset: args.preset,
        sample_rate: args.sample_rate,
        descriptor,
        warnings: analysis.warnings.clone(),
        generated_at_utc: Utc::now(),
        analysis,
    };

    let metadata_path = args
        .metadata_out
        .unwrap_or_else(|| util::metadata::companion_json_path(&args.output));
    util::json::write_pretty_json(&metadata_path, &metadata)?;

    if let Some(analysis_path) = args.json_analysis_out {
        util::json::write_pretty_json(&analysis_path, &metadata.analysis)?;
        println!("wrote analysis: {}", analysis_path.display());
    }

    println!("wrote IR: {}", args.output.display());
    println!("wrote metadata: {}", metadata_path.display());
    Ok(())
}
