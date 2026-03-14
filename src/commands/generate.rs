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
    validate_generate_args(&args)?;
    let mut descriptor = DescriptorSet::default();
    let mut conditioning = ConditioningTrace::default();
    let mut runtime_warnings = Vec::new();

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
    let pre_clamp = descriptor.clone();
    descriptor.clamp();
    runtime_warnings.extend(descriptor_clamp_warnings(&pre_clamp, &descriptor));
    apply_duration_floor(
        &mut descriptor,
        args.allow_tail_truncation,
        &mut runtime_warnings,
    );

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

    let mut analysis = IrAnalyzer::default().analyze_with_channel_map(
        &generated.channels,
        args.sample_rate,
        Some(&channel_map),
    );
    analysis.warnings.extend(runtime_warnings);
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

fn validate_generate_args(args: &GenerateArgs) -> Result<()> {
    anyhow::ensure!(
        (8_000..=768_000).contains(&args.sample_rate),
        "sample rate {} out of supported range [8000, 768000] Hz",
        args.sample_rate
    );
    ensure_positive_finite("duration", args.duration, false)?;
    ensure_positive_finite("t60", args.t60, false)?;
    ensure_positive_finite("predelay-ms", args.predelay_ms, true)?;
    ensure_positive_finite("edt", args.edt, false)?;
    Ok(())
}

fn ensure_positive_finite(name: &str, v: Option<f32>, allow_zero: bool) -> Result<()> {
    if let Some(v) = v {
        anyhow::ensure!(v.is_finite(), "{name} must be finite");
        if allow_zero {
            anyhow::ensure!(v >= 0.0, "{name} must be >= 0");
        } else {
            anyhow::ensure!(v > 0.0, "{name} must be > 0");
        }
    }
    Ok(())
}

fn descriptor_clamp_warnings(before: &DescriptorSet, after: &DescriptorSet) -> Vec<String> {
    let mut out = Vec::new();
    push_if_changed(
        &mut out,
        "duration_s",
        before.time.duration,
        after.time.duration,
    );
    push_if_changed(
        &mut out,
        "predelay_ms",
        before.time.predelay_ms,
        after.time.predelay_ms,
    );
    push_if_changed(&mut out, "t60_s", before.time.t60, after.time.t60);
    push_if_changed(&mut out, "edt_s", before.time.edt, after.time.edt);
    out
}

fn push_if_changed(out: &mut Vec<String>, label: &str, before: f32, after: f32) {
    if (before - after).abs() > 1e-6 {
        out.push(format!(
            "descriptor '{label}' was clamped from {before:.4} to {after:.4}"
        ));
    }
}

fn apply_duration_floor(
    descriptor: &mut DescriptorSet,
    allow_tail_truncation: bool,
    warnings: &mut Vec<String>,
) {
    let predelay_s = descriptor.time.predelay_ms.max(0.0) * 0.001;
    let recommended = (predelay_s
        + (descriptor.time.t60 * 1.15).max(descriptor.time.edt * 1.4)
        + descriptor.time.attack_gap_ms.max(0.0) * 0.001
        + 0.05)
        .clamp(0.1, 30.0);

    if descriptor.time.duration + 1e-6 >= recommended {
        return;
    }

    if allow_tail_truncation {
        warnings.push(format!(
            "duration {:.3}s is below recommended {:.3}s for current decay settings; tail may truncate (--allow-tail-truncation enabled)",
            descriptor.time.duration, recommended
        ));
        return;
    }

    // Default to preserving decay intent; hard cuts are great for drums, not so much for cathedral IRs.
    let original = descriptor.time.duration;
    descriptor.time.duration = recommended;
    if (recommended - 30.0).abs() < 1e-6 {
        warnings.push(format!(
            "duration auto-extended from {:.3}s to {:.3}s (maximum supported); long T60 may still truncate near output end",
            original, recommended
        ));
    } else {
        warnings.push(format!(
            "duration auto-extended from {:.3}s to {:.3}s to better capture requested decay",
            original, recommended
        ));
    }
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
        (Some(x), Some(y), Some(z)) => {
            anyhow::ensure!(
                x.is_finite() && y.is_finite() && z.is_finite(),
                "{prefix} position coordinates must be finite numbers"
            );
            Ok(Some(CartesianPosition { x, y, z }))
        }
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
        util::console::metric_opt(
            "inter_channel_corr_mean_abs",
            fmt_opt(r.inter_channel_correlation_mean_abs, 5)
        )
    );
    println!(
        "{}",
        util::console::metric_opt(
            "inter_channel_corr_min_abs",
            fmt_opt(r.inter_channel_correlation_min_abs, 5)
        )
    );
    println!(
        "{}",
        util::console::metric_opt("arrival_min_ms", fmt_opt(r.arrival_min_ms, 3))
    );
    println!(
        "{}",
        util::console::metric_opt("arrival_max_ms", fmt_opt(r.arrival_max_ms, 3))
    );
    println!(
        "{}",
        util::console::metric_opt("arrival_spread_ms", fmt_opt(r.arrival_spread_ms, 3))
    );
    println!(
        "{}",
        util::console::metric_opt("itd_01_ms", fmt_opt(r.itd_01_ms, 3))
    );
    println!(
        "{}",
        util::console::metric_opt("iacc_early_01", fmt_opt(r.iacc_early_01, 5))
    );
    println!(
        "{}",
        util::console::metric_opt(
            "inter_channel_itd_mean_abs_ms",
            fmt_opt(r.inter_channel_itd_mean_abs_ms, 3)
        )
    );
    println!(
        "{}",
        util::console::metric_opt(
            "inter_channel_itd_max_abs_ms",
            fmt_opt(r.inter_channel_itd_max_abs_ms, 3)
        )
    );
    println!(
        "{}",
        util::console::metric_opt(
            "inter_channel_iacc_early_mean",
            fmt_opt(r.inter_channel_iacc_early_mean, 5)
        )
    );
    println!(
        "{}",
        util::console::metric_opt(
            "inter_channel_iacc_early_min",
            fmt_opt(r.inter_channel_iacc_early_min, 5)
        )
    );
    println!(
        "{}",
        util::console::metric_opt("front_energy_ratio", fmt_opt(r.front_energy_ratio, 5))
    );
    println!(
        "{}",
        util::console::metric_opt("rear_energy_ratio", fmt_opt(r.rear_energy_ratio, 5))
    );
    println!(
        "{}",
        util::console::metric_opt("height_energy_ratio", fmt_opt(r.height_energy_ratio, 5))
    );
    println!(
        "{}",
        util::console::metric_opt("lfe_energy_ratio", fmt_opt(r.lfe_energy_ratio, 5))
    );
    println!(
        "{}",
        util::console::metric_opt("stereo_correlation", fmt_opt(r.stereo_correlation, 5))
    );
    if !r.warnings.is_empty() {
        println!("{}", util::console::warning("warnings:"));
        for w in &r.warnings {
            println!("  {}", util::console::warning(&format!("- {w}")));
        }
    }
    println!("{}", util::console::section("----------------------------"));
}

fn fmt_opt(value: Option<f32>, decimals: usize) -> Option<String> {
    value.map(|v| format!("{:.*}", decimals, v))
}
