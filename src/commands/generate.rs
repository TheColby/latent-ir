use anyhow::{Context, Result};
use chrono::Utc;

use crate::cli::{ChannelFormatArg, GenerateArgs, QualityProfileArg};
use crate::core::analysis::{evaluate_quality_gate, AnalysisReport, IrAnalyzer, QualityProfile};
use crate::core::conditioning::{
    estimate_conditioning_uncertainty, run_conditioning_chain, ConditioningContext,
    ConditioningModel, DescriptorDelta, JsonAudioConditioningModel, JsonTextConditioningModel,
    LearnedAudioEncoder, LearnedTextEncoder, OnnxAudioConditioningModel, OnnxTextConditioningModel,
    SemanticConditioningModel,
};
use crate::core::descriptors::{CartesianPosition, ChannelFormat, DescriptorSet};
use crate::core::generator::{generate_with_macro_trajectory, IrGenerator, ProceduralIrGenerator};
use crate::core::perceptual::{MacroControls, MacroTrajectory};
use crate::core::presets;
use crate::core::semantics::channel_format_hint;
use crate::core::spatial;
use crate::core::util::{
    self,
    metadata::{ConditioningTrace, GenerationMetadata},
};

pub fn run(args: GenerateArgs) -> Result<()> {
    validate_generate_args(&args)?;
    let replay_command = build_replay_command(&args);
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
    conditioning.uncertainty = estimate_conditioning_uncertainty(&chain_out.by_model);
    let combined_delta = chain_out.total_delta.clone();
    combined_delta.apply_to(&mut descriptor, 1.0);
    conditioning.combined_delta = Some(combined_delta);

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

    let wants_custom_layout =
        resolve_custom_layout_intent(args.channels, args.layout_json.is_some())?;
    if let Some(layout_path) = args.layout_json.as_deref() {
        let layout = spatial::load_custom_layout_file(layout_path)?;
        descriptor.spatial.set_custom_layout(layout);
    } else if wants_custom_layout {
        anyhow::bail!("--channels custom requires --layout-json <path>");
    }

    descriptor.apply_overrides(args.duration, args.t60, args.predelay_ms, args.edt);
    descriptor.apply_spectral_overrides(args.brightness, None, None, None);
    descriptor.apply_structure_overrides(args.early_density, args.late_density, args.diffusion);
    let semantic_channel_hint = if args.channels.is_none() && args.layout_json.is_none() {
        args.prompt.as_deref().and_then(channel_format_hint)
    } else {
        None
    };
    descriptor.apply_spatial_overrides(
        args.channels
            .map(channel_format_from_arg)
            .or(semantic_channel_hint),
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
    let mut generated = if let Some(traj) = trajectory.as_ref() {
        generate_with_macro_trajectory(&generator, &descriptor, traj, args.seed)?
    } else {
        generator.generate(&descriptor, args.seed)?
    };
    if let Some(tail_fade_ms) = args.tail_fade_ms {
        if let Some(w) =
            util::audio::apply_tail_fade(&mut generated.channels, args.sample_rate, tail_fade_ms)
        {
            runtime_warnings.push(w);
        }
    }

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
    let quality_gate_result = if args.quality_gate {
        Some(evaluate_quality_gate(
            &analysis,
            quality_profile_from_arg(args.quality_profile),
        ))
    } else {
        None
    };
    if let Some(gate) = quality_gate_result.as_ref() {
        if !gate.passed {
            analysis.warnings.push(format!(
                "quality gate '{}' failed ({} checks)",
                format!("{:?}", gate.profile).to_lowercase(),
                gate.failed_checks.len()
            ));
        }
    }
    let channel_labels = descriptor.spatial.resolved_channel_labels();
    let ir_sha256 = util::hash::sha256_channels_f32(&generated.channels);
    let descriptor_sha256 = util::hash::sha256_json(&descriptor)?;
    let channel_map_sha256 = util::hash::sha256_json(&channel_map)?;

    let metadata = GenerationMetadata {
        schema_version: "latent-ir.generation.v1".to_string(),
        project: "latent-ir".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        command: "generate".to_string(),
        replay_command,
        seed: args.seed,
        prompt: args.prompt,
        preset: args.preset,
        conditioning,
        sample_rate: args.sample_rate,
        spatial_encoding: channel_map.spatial_encoding.clone(),
        channel_format: channel_map.layout_name.clone(),
        channel_labels,
        channel_map_path: Some(channel_map_path.display().to_string()),
        ir_sha256,
        descriptor_sha256,
        channel_map_sha256,
        descriptor,
        quality_gate_profile: quality_gate_result
            .as_ref()
            .map(|g| format!("{:?}", g.profile).to_lowercase()),
        quality_gate_passed: quality_gate_result.as_ref().map(|g| g.passed),
        quality_gate_failed_checks: quality_gate_result
            .as_ref()
            .map(|g| g.failed_checks.clone()),
        warnings: analysis.warnings.clone(),
        generated_at_utc: Utc::now(),
        analysis,
    };

    let metadata_path = args
        .metadata_out
        .unwrap_or_else(|| util::metadata::companion_json_path(&args.output));
    util::json::write_pretty_json(&metadata_path, &metadata)?;

    if args.explain_conditioning {
        print_conditioning_summary(&metadata.conditioning);
        print_descriptor_snapshot(&metadata.descriptor);
    }

    if let Some(analysis_path) = args.json_analysis_out {
        util::json::write_pretty_json(&analysis_path, &metadata.analysis)?;
        println!("wrote analysis: {}", analysis_path.display());
    }

    print_generation_metrics(
        &metadata.analysis,
        &metadata.channel_format,
        &metadata.channel_labels,
    );
    println!(
        "{}",
        util::console::metric("ir_sha256", &metadata.ir_sha256)
    );
    println!(
        "{}",
        util::console::metric("descriptor_sha256", &metadata.descriptor_sha256)
    );
    println!(
        "{}",
        util::console::metric("channel_map_sha256", &metadata.channel_map_sha256)
    );
    if let Some(gate) = quality_gate_result.as_ref() {
        print_quality_gate(gate);
        if !gate.passed {
            anyhow::bail!(
                "quality gate failed for profile '{}' ({} checks)",
                format!("{:?}", gate.profile).to_lowercase(),
                gate.failed_checks.len()
            );
        }
    }
    println!("wrote IR: {}", args.output.display());
    println!("wrote metadata: {}", metadata_path.display());
    println!("wrote channel map: {}", channel_map_path.display());
    Ok(())
}

fn build_replay_command(args: &GenerateArgs) -> String {
    let mut parts = vec!["latent-ir".to_string(), "generate".to_string()];
    if args.explain_conditioning {
        parts.push("--explain-conditioning".to_string());
    }
    if let Some(v) = args.prompt.as_deref() {
        parts.push("--prompt".to_string());
        parts.push(shell_quote(v));
    }
    if let Some(v) = args.text_encoder_model.as_deref() {
        parts.push("--text-encoder-model".to_string());
        parts.push(shell_quote(&v.display().to_string()));
    }
    if let Some(v) = args.audio_encoder_model.as_deref() {
        parts.push("--audio-encoder-model".to_string());
        parts.push(shell_quote(&v.display().to_string()));
    }
    if let Some(v) = args.reference_audio.as_deref() {
        parts.push("--reference-audio".to_string());
        parts.push(shell_quote(&v.display().to_string()));
    }
    if let Some(v) = args.preset.as_deref() {
        parts.push("--preset".to_string());
        parts.push(shell_quote(v));
    }
    if let Some(v) = args.duration {
        parts.push("--duration".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.t60 {
        parts.push("--t60".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.predelay_ms {
        parts.push("--predelay-ms".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.edt {
        parts.push("--edt".to_string());
        parts.push(format!("{v}"));
    }
    parts.push("--sample-rate".to_string());
    parts.push(format!("{}", args.sample_rate));
    parts.push("--seed".to_string());
    parts.push(format!("{}", args.seed));
    if args.allow_tail_truncation {
        parts.push("--allow-tail-truncation".to_string());
    }
    if let Some(v) = args.tail_fade_ms {
        parts.push("--tail-fade-ms".to_string());
        parts.push(format!("{v}"));
    }
    if args.quality_gate {
        parts.push("--quality-gate".to_string());
        parts.push("--quality-profile".to_string());
        parts.push(quality_profile_arg_value(args.quality_profile).to_string());
    }
    if let Some(v) = args.layout_json.as_deref() {
        parts.push("--layout-json".to_string());
        parts.push(shell_quote(&v.display().to_string()));
    }
    if let Some(v) = args.source_x_m {
        parts.push("--source-x-m".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.source_y_m {
        parts.push("--source-y-m".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.source_z_m {
        parts.push("--source-z-m".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.listener_x_m {
        parts.push("--listener-x-m".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.listener_y_m {
        parts.push("--listener-y-m".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(v) = args.listener_z_m {
        parts.push("--listener-z-m".to_string());
        parts.push(format!("{v}"));
    }
    if let Some(ch) = args.channels {
        parts.push("--channels".to_string());
        parts.push(shell_quote(channel_arg_value(ch)));
    } else if args.layout_json.is_some() {
        parts.push("--channels".to_string());
        parts.push("custom".to_string());
    }
    parts.push("--output".to_string());
    parts.push(shell_quote(&args.output.display().to_string()));
    parts.join(" ")
}

fn channel_arg_value(arg: ChannelFormatArg) -> &'static str {
    match arg {
        ChannelFormatArg::Mono => "mono",
        ChannelFormatArg::Stereo => "stereo",
        ChannelFormatArg::Foa => "foa",
        ChannelFormatArg::Surround5_1 => "5.1",
        ChannelFormatArg::Surround7_1 => "7.1",
        ChannelFormatArg::Atmos7_1_4 => "7.1.4",
        ChannelFormatArg::Atmos7_2_4 => "7.2.4",
        ChannelFormatArg::Custom => "custom",
    }
}

fn resolve_custom_layout_intent(
    channels: Option<ChannelFormatArg>,
    has_layout_json: bool,
) -> Result<bool> {
    match (channels, has_layout_json) {
        (Some(ChannelFormatArg::Custom), true) => Ok(true),
        (Some(ChannelFormatArg::Custom), false) => Ok(true),
        (Some(_), true) => anyhow::bail!("--layout-json is only valid with --channels custom"),
        (Some(_), false) => Ok(false),
        (None, true) => Ok(true),
        (None, false) => Ok(false),
    }
}

fn shell_quote(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/')
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn print_conditioning_summary(c: &ConditioningTrace) {
    println!("{}", util::console::section("--- conditioning summary ---"));
    println!(
        "{}",
        util::console::metric(
            "text_encoder_model",
            c.text_encoder_model.as_deref().unwrap_or("n/a")
        )
    );
    println!(
        "{}",
        util::console::metric(
            "audio_encoder_model",
            c.audio_encoder_model.as_deref().unwrap_or("n/a")
        )
    );
    print_delta("combined_delta", c.combined_delta.as_ref());
    print_delta("text_delta", c.text_delta.as_ref());
    print_delta("audio_delta", c.audio_delta.as_ref());
    if let Some(u) = c.uncertainty.as_ref() {
        println!(
            "{}",
            util::console::metric(
                "conditioning_confidence",
                format!("{:.3}", u.overall_confidence)
            )
        );
        println!(
            "{}",
            util::console::metric(
                "conditioning_uncertainty",
                format!("{:.3}", u.overall_uncertainty)
            )
        );
        println!(
            "{}",
            util::console::metric(
                "conditioning_agreement",
                format!("{:.3}", u.agreement_score)
            )
        );
    } else {
        println!(
            "{}",
            util::console::metric("conditioning_confidence", "n/a")
        );
    }
    println!("{}", util::console::section("----------------------------"));
}

fn print_delta(name: &str, delta: Option<&DescriptorDelta>) {
    let Some(d) = delta else {
        println!("{}", util::console::metric(name, "n/a"));
        return;
    };
    println!(
        "{}",
        util::console::metric(
            name,
            format!(
                "t60={:+.3}, predelay_ms={:+.3}, brightness={:+.3}, diffusion={:+.3}, width={:+.3}",
                d.t60, d.predelay_ms, d.brightness, d.diffusion, d.width
            )
        )
    );
}

fn print_descriptor_snapshot(d: &DescriptorSet) {
    println!("{}", util::console::section("--- resolved descriptor ---"));
    println!(
        "{}",
        util::console::metric(
            "time",
            format!(
                "duration={:.3}s, t60={:.3}s, predelay={:.2}ms, edt={:.3}s",
                d.time.duration, d.time.t60, d.time.predelay_ms, d.time.edt
            )
        )
    );
    println!(
        "{}",
        util::console::metric(
            "spectral",
            format!(
                "brightness={:.3}, hf_damping={:.3}, lf_bloom={:.3}",
                d.spectral.brightness, d.spectral.hf_damping, d.spectral.lf_bloom
            )
        )
    );
    println!(
        "{}",
        util::console::metric(
            "structural",
            format!(
                "early_density={:.3}, late_density={:.3}, diffusion={:.3}",
                d.structural.early_density, d.structural.late_density, d.structural.diffusion
            )
        )
    );
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
    ensure_positive_finite("tail-fade-ms", args.tail_fade_ms, true)?;
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

fn quality_profile_arg_value(arg: QualityProfileArg) -> &'static str {
    match arg {
        QualityProfileArg::Lenient => "lenient",
        QualityProfileArg::Launch => "launch",
        QualityProfileArg::Strict => "strict",
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
        util::console::metric("decay_db_span", format!("{:.2}", r.decay_db_span))
    );
    println!(
        "{}",
        util::console::metric_opt("t60_confidence", fmt_opt(r.t60_confidence, 3))
    );
    println!(
        "{}",
        util::console::metric_opt("edt_confidence", fmt_opt(r.edt_confidence, 3))
    );
    println!(
        "{}",
        util::console::metric("crest_factor_db", format!("{:.2}", r.crest_factor_db))
    );
    println!(
        "{}",
        util::console::metric_opt(
            "tail_reaches_minus60db_s",
            fmt_opt(r.tail_reaches_minus60db_s, 3)
        )
    );
    println!(
        "{}",
        util::console::metric_opt("tail_margin_to_end_s", fmt_opt(r.tail_margin_to_end_s, 3))
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

fn print_quality_gate(gate: &crate::core::analysis::QualityGateResult) {
    println!("{}", util::console::section("--- quality gate ---"));
    println!(
        "{}",
        util::console::metric("profile", format!("{:?}", gate.profile).to_lowercase())
    );
    println!("{}", util::console::metric("passed", gate.passed));
    if !gate.failed_checks.is_empty() {
        println!("{}", util::console::warning("failed_checks:"));
        for check in &gate.failed_checks {
            println!("  {}", util::console::warning(&format!("- {check}")));
        }
    }
    println!("{}", util::console::section("--------------------"));
}

fn quality_profile_from_arg(arg: QualityProfileArg) -> QualityProfile {
    match arg {
        QualityProfileArg::Lenient => QualityProfile::Lenient,
        QualityProfileArg::Launch => QualityProfile::Launch,
        QualityProfileArg::Strict => QualityProfile::Strict,
    }
}

fn fmt_opt(value: Option<f32>, decimals: usize) -> Option<String> {
    value.map(|v| format!("{:.*}", decimals, v))
}
