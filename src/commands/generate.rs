use anyhow::{Context, Result};
use chrono::Utc;

use crate::cli::{ChannelFormatArg, GenerateArgs};
use crate::core::analysis::IrAnalyzer;
use crate::core::conditioning::{
    run_conditioning_chain, ConditioningContext, ConditioningModel, JsonAudioConditioningModel,
    JsonTextConditioningModel, LearnedAudioEncoder, LearnedTextEncoder, OnnxAudioConditioningModel,
    OnnxTextConditioningModel, SemanticConditioningModel,
};
use crate::core::descriptors::{ChannelFormat, DescriptorSet};
use crate::core::generator::{generate_with_macro_trajectory, IrGenerator, ProceduralIrGenerator};
use crate::core::perceptual::{MacroControls, MacroTrajectory};
use crate::core::presets;
use crate::core::util::{
    self,
    metadata::{ConditioningTrace, GenerationMetadata},
};

pub fn run(args: GenerateArgs) -> Result<()> {
    let mut descriptor = DescriptorSet::default();
    let mut conditioning = ConditioningTrace::default();

    if let Some(name) = args.preset.as_deref() {
        descriptor =
            presets::resolve_preset(name).with_context(|| format!("unknown preset '{name}'"))?;
    }

    let reference_audio = if let Some(path) = args.reference_audio.as_deref() {
        Some(
            util::audio::read_wav_f32(path)
                .with_context(|| format!("failed to read {}", path.display()))?,
        )
    } else {
        None
    };
    if let Some(path) = args.reference_audio.as_deref() {
        conditioning.reference_audio = Some(path.display().to_string());
    }

    let mut chain: Vec<Box<dyn ConditioningModel>> = vec![Box::new(SemanticConditioningModel)];
    if let Some(model_path) = args.text_encoder_model.as_deref() {
        let model = LearnedTextEncoder::from_json_file(model_path)?;
        chain.push(Box::new(JsonTextConditioningModel {
            source_path: model_path.to_path_buf(),
            model,
        }));
    }
    if let Some(model_path) = args.audio_encoder_model.as_deref() {
        let model = LearnedAudioEncoder::from_json_file(model_path)?;
        chain.push(Box::new(JsonAudioConditioningModel {
            source_path: model_path.to_path_buf(),
            model,
        }));
    }
    if let Some(model_path) = args.text_encoder_onnx.as_deref() {
        chain.push(Box::new(OnnxTextConditioningModel {
            source_path: model_path.to_path_buf(),
        }));
    }
    if let Some(model_path) = args.audio_encoder_onnx.as_deref() {
        chain.push(Box::new(OnnxAudioConditioningModel {
            source_path: model_path.to_path_buf(),
        }));
    }

    let ctx = ConditioningContext {
        prompt: args.prompt.clone(),
        reference_audio,
        text_onnx_input_dim: args.text_encoder_onnx_input_dim,
    };
    let chain_out = run_conditioning_chain(&chain, &ctx)?;
    chain_out.total_delta.apply_to(&mut descriptor, 1.0);

    for (name, delta) in chain_out.by_model {
        match name.as_str() {
            "text_json" => {
                conditioning.text_encoder_model = args
                    .text_encoder_model
                    .as_ref()
                    .map(|p| p.display().to_string());
                conditioning.text_delta = Some(delta);
            }
            "text_onnx" => {
                conditioning.text_encoder_onnx = args
                    .text_encoder_onnx
                    .as_ref()
                    .map(|p| p.display().to_string());
                if let Some(existing) = conditioning.text_delta.as_mut() {
                    existing.add_inplace(&delta);
                } else {
                    conditioning.text_delta = Some(delta);
                }
            }
            "audio_json" => {
                conditioning.audio_encoder_model = args
                    .audio_encoder_model
                    .as_ref()
                    .map(|p| p.display().to_string());
                conditioning.audio_delta = Some(delta);
            }
            "audio_onnx" => {
                conditioning.audio_encoder_onnx = args
                    .audio_encoder_onnx
                    .as_ref()
                    .map(|p| p.display().to_string());
                if let Some(existing) = conditioning.audio_delta.as_mut() {
                    existing.add_inplace(&delta);
                } else {
                    conditioning.audio_delta = Some(delta);
                }
            }
            _ => {}
        }
    }

    let mut macros = MacroControls {
        size: args.macro_size.unwrap_or(0.0),
        distance: args.macro_distance.unwrap_or(0.0),
        material: args.macro_material.unwrap_or(0.0),
        clarity: args.macro_clarity.unwrap_or(0.0),
    };
    macros.clamp();
    if macros != MacroControls::default() {
        macros.apply_to(&mut descriptor);
        conditioning.macro_controls = Some(macros.clone());
    }

    let trajectory = if let Some(path) = args.macro_trajectory.as_deref() {
        let traj = MacroTrajectory::from_json_file(path)?;
        conditioning.macro_trajectory = Some(path.display().to_string());
        traj.apply_static_average(&mut descriptor);
        Some(traj)
    } else {
        None
    };

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
    let generated = if let Some(traj) = trajectory.as_ref() {
        generate_with_macro_trajectory(&generator, &descriptor, traj, args.seed)?
    } else {
        generator.generate(&descriptor, args.seed)?
    };

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
        conditioning,
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
