use anyhow::{Context, Result};
use chrono::Utc;

use crate::cli::{ChannelFormatArg, GenerateArgs};
use crate::core::analysis::{AnalysisReport, IrAnalyzer};
use crate::core::conditioning::{
    run_conditioning_chain, ConditioningContext, ConditioningModel, JsonAudioConditioningModel,
    JsonTextConditioningModel, LearnedAudioEncoder, LearnedTextEncoder, OnnxAudioConditioningModel,
    OnnxTextConditioningModel, SemanticConditioningModel,
};
use crate::core::descriptors::{CartesianPosition, ChannelFormat, DescriptorSet};
use crate::core::generator::{generate_with_macro_trajectory, IrGenerator, ProceduralIrGenerator};
use crate::core::perceptual::{MacroControls, MacroTrajectory};
use crate::core::presets;
use crate::core::spatial;
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

    spatial::ensure_custom_layout_requested(
        args.channels == ChannelFormatArg::Custom,
        args.layout_json.is_some(),
    )?;
    if let Some(layout_path) = args.layout_json.as_deref() {
        let layout = spatial::load_custom_layout_file(layout_path)?;
        descriptor.spatial.set_custom_layout(layout);
    }

    descriptor.apply_overrides(args.duration, args.t60, args.predelay_ms, args.edt);
    descriptor.apply_spectral_overrides(args.brightness, None, None, None);
    descriptor.apply_structure_overrides(args.early_density, args.late_density, args.diffusion);
    descriptor.apply_spatial_overrides(
        Some(channel_format_from_arg(args.channels)),
        args.width,
        args.decorrelation,
        None,
    );

    descriptor.spatial.source_position_m =
        parse_position_triplet("source", args.source_x_m, args.source_y_m, args.source_z_m)?;
    descriptor.spatial.listener_position_m = parse_position_triplet(
        "listener",
        args.listener_x_m,
        args.listener_y_m,
        args.listener_z_m,
    )?;
    descriptor.clamp();

    let generator = ProceduralIrGenerator::new(args.sample_rate);
    let generated = if let Some(traj) = trajectory.as_ref() {
        generate_with_macro_trajectory(&generator, &descriptor, traj, args.seed)?
    } else {
        generator.generate(&descriptor, args.seed)?
    };

    util::audio::write_wav_f32(&args.output, args.sample_rate, &generated.channels)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    let channel_map = spatial::build_channel_map(&descriptor.spatial);
    spatial::validate_channel_map(&channel_map, generated.channels.len())?;
    let channel_map_path = args
        .channel_map_out
        .unwrap_or_else(|| spatial::companion_channel_map_path(&args.output));
    util::json::write_pretty_json(&channel_map_path, &channel_map)?;

    let analysis = IrAnalyzer::default().analyze_with_channel_map(
        &generated.channels,
        args.sample_rate,
        Some(&channel_map),
    );
    let channel_labels = descriptor.spatial.resolved_channel_labels();

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
        spatial_encoding: channel_map.spatial_encoding.clone(),
        channel_format: channel_map.layout_name.clone(),
        channel_labels,
        channel_map_path: Some(channel_map_path.display().to_string()),
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

    print_generation_metrics(
        &metadata.analysis,
        &metadata.channel_format,
        &metadata.channel_labels,
    );
    println!("wrote IR: {}", args.output.display());
    println!("wrote metadata: {}", metadata_path.display());
    println!("wrote channel map: {}", channel_map_path.display());
    Ok(())
}

fn channel_format_from_arg(arg: ChannelFormatArg) -> ChannelFormat {
    match arg {
        ChannelFormatArg::Mono => ChannelFormat::Mono,
        ChannelFormatArg::Stereo => ChannelFormat::Stereo,
        ChannelFormatArg::Foa => ChannelFormat::FoaAmbix,
        ChannelFormatArg::Surround5_1 => ChannelFormat::Surround5_1,
        ChannelFormatArg::Surround7_1 => ChannelFormat::Surround7_1,
        ChannelFormatArg::Atmos7_1_4 => ChannelFormat::Atmos7_1_4,
        ChannelFormatArg::Atmos7_2_4 => ChannelFormat::Atmos7_2_4,
        ChannelFormatArg::Custom => ChannelFormat::Custom,
    }
}

fn parse_position_triplet(
    prefix: &str,
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
) -> Result<Option<CartesianPosition>> {
    match (x, y, z) {
        (None, None, None) => Ok(None),
        (Some(x), Some(y), Some(z)) => Ok(Some(CartesianPosition { x, y, z })),
        _ => anyhow::bail!(
            "{prefix} position requires all three coordinates: --{prefix}-x-m --{prefix}-y-m --{prefix}-z-m"
        ),
    }
}

fn print_generation_metrics(r: &AnalysisReport, channel_format: &str, channel_labels: &[String]) {
    println!("{}", util::console::section("--- generated IR metrics ---"));
    println!(
        "{}",
        util::console::warning(
            "metrics_note: engineering estimates for v0 (not standards-certified architectural acoustics metrology)"
        )
    );
    println!("{}", util::console::metric("sample_rate_hz", r.sample_rate));
    println!(
        "{}",
        util::console::metric("channel_format", channel_format)
    );
    println!(
        "{}",
        util::console::metric("channel_labels", channel_labels.join(","))
    );
    println!("{}", util::console::metric("channels", r.channels));
    println!(
        "{}",
        util::console::metric("ir_length_s", format!("{:.4}", r.duration_s))
    );
    println!(
        "{}",
        util::console::metric("peak", format!("{:.6}", r.peak))
    );
    println!("{}", util::console::metric("rms", format!("{:.6}", r.rms)));
    println!(
        "{}",
        util::console::metric("predelay_ms_est", format!("{:.3}", r.predelay_ms_est))
    );
    println!(
        "{}",
        util::console::metric("edt_s_est", format!("{:.4}", r.edt_s_est.unwrap_or(-1.0)))
    );
    println!(
        "{}",
        util::console::metric("t20_s_est", format!("{:.4}", r.t20_s_est.unwrap_or(-1.0)))
    );
    println!(
        "{}",
        util::console::metric("t30_s_est", format!("{:.4}", r.t30_s_est.unwrap_or(-1.0)))
    );
    println!(
        "{}",
        util::console::metric("t60_s_est", format!("{:.4}", r.t60_s_est.unwrap_or(-1.0)))
    );
    println!(
        "{}",
        util::console::metric(
            "spectral_centroid_hz",
            format!("{:.2}", r.spectral_centroid_hz)
        )
    );
    println!(
        "{}",
        util::console::metric(
            "band_decay_low_s",
            format!("{:.4}", r.band_decay_low_s.unwrap_or(-1.0))
        )
    );
    println!(
        "{}",
        util::console::metric(
            "band_decay_mid_s",
            format!("{:.4}", r.band_decay_mid_s.unwrap_or(-1.0))
        )
    );
    println!(
        "{}",
        util::console::metric(
            "band_decay_high_s",
            format!("{:.4}", r.band_decay_high_s.unwrap_or(-1.0))
        )
    );
    println!(
        "{}",
        util::console::metric("early_energy_ratio", format!("{:.5}", r.early_energy_ratio))
    );
    println!(
        "{}",
        util::console::metric("late_energy_ratio", format!("{:.5}", r.late_energy_ratio))
    );
    println!(
        "{}",
        util::console::metric(
            "inter_channel_corr_mean_abs",
            match r.inter_channel_correlation_mean_abs {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "inter_channel_corr_min_abs",
            match r.inter_channel_correlation_min_abs {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "arrival_min_ms",
            match r.arrival_min_ms {
                Some(v) => format!("{v:.3}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "arrival_max_ms",
            match r.arrival_max_ms {
                Some(v) => format!("{v:.3}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "arrival_spread_ms",
            match r.arrival_spread_ms {
                Some(v) => format!("{v:.3}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "itd_01_ms",
            match r.itd_01_ms {
                Some(v) => format!("{v:.3}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "iacc_early_01",
            match r.iacc_early_01 {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "inter_channel_itd_mean_abs_ms",
            match r.inter_channel_itd_mean_abs_ms {
                Some(v) => format!("{v:.3}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "inter_channel_itd_max_abs_ms",
            match r.inter_channel_itd_max_abs_ms {
                Some(v) => format!("{v:.3}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "inter_channel_iacc_early_mean",
            match r.inter_channel_iacc_early_mean {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "inter_channel_iacc_early_min",
            match r.inter_channel_iacc_early_min {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "front_energy_ratio",
            match r.front_energy_ratio {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "rear_energy_ratio",
            match r.rear_energy_ratio {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "height_energy_ratio",
            match r.height_energy_ratio {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "lfe_energy_ratio",
            match r.lfe_energy_ratio {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    println!(
        "{}",
        util::console::metric(
            "stereo_correlation",
            match r.stereo_correlation {
                Some(v) => format!("{v:.5}"),
                None => "n/a".to_string(),
            }
        )
    );
    if !r.warnings.is_empty() {
        println!("{}", util::console::warning("warnings:"));
        for w in &r.warnings {
            println!("  {}", util::console::warning(&format!("- {w}")));
        }
    }
    println!("{}", util::console::section("----------------------------"));
}
