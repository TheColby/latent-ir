use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::cli::{
    ChannelFormatArg, DatasetArgs, DatasetMode, DatasetSplitArgs, DatasetSynthesizeArgs,
    QualityProfileArg,
};
use crate::core::analysis::{evaluate_quality_gate, IrAnalyzer, QualityProfile};
use crate::core::dataset::{
    DatasetConfigSnapshot, DatasetManifest, DatasetRecord, DatasetSplitCounts,
    DatasetSplitManifest, DatasetSplitRatios, DatasetSplitRecord, DatasetSummary,
};
use crate::core::descriptors::{ChannelFormat, DescriptorSet};
use crate::core::generator::{IrGenerator, ProceduralIrGenerator};
use crate::core::presets;
use crate::core::semantics::SemanticResolver;
use crate::core::spatial;
use crate::core::training::{AudioTrainSample, TextTrainSample};
use crate::core::util::{
    self,
    metadata::{ConditioningTrace, GenerationMetadata},
};

pub fn run(args: DatasetArgs) -> Result<()> {
    match args.mode {
        DatasetMode::Synthesize(cfg) => run_synthesize(cfg),
        DatasetMode::Split(cfg) => run_split(cfg),
    }
}

fn run_synthesize(args: DatasetSynthesizeArgs) -> Result<()> {
    validate_args(&args)?;
    let prompt_bank = load_prompt_bank(args.prompt_bank_json.as_deref())?;
    let preset_names = presets::preset_names();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let quality_profile = quality_profile_from_arg(args.quality_profile);

    let ir_dir = args.out_dir.join("ir");
    let metadata_dir = args.out_dir.join("metadata");
    let analysis_dir = args.out_dir.join("analysis");
    let channel_map_dir = args.out_dir.join("channel_maps");
    std::fs::create_dir_all(&ir_dir)?;
    std::fs::create_dir_all(&metadata_dir)?;
    std::fs::create_dir_all(&analysis_dir)?;
    std::fs::create_dir_all(&channel_map_dir)?;

    let generator = ProceduralIrGenerator::new(args.sample_rate);
    let analyzer = IrAnalyzer::default();
    let semantic = SemanticResolver;

    let mut records = Vec::with_capacity(args.count);
    let mut text_train = Vec::new();
    let mut audio_train = Vec::new();
    let mut failures = Vec::new();

    for idx in 0..args.count {
        let id = format!("sample_{idx:05}");
        let sample_seed = args.seed.wrapping_add((idx as u64).wrapping_mul(7_919));

        let prompt = prompt_bank[rng.gen_range(0..prompt_bank.len())].clone();
        let preset = if rng.gen_bool(args.preset_mix.clamp(0.0, 1.0) as f64) {
            Some(preset_names[rng.gen_range(0..preset_names.len())].to_string())
        } else {
            None
        };

        let mut descriptor = if let Some(name) = preset.as_deref() {
            presets::resolve_preset(name)?
        } else {
            DescriptorSet::default()
        };
        semantic.apply_prompt(&prompt, &mut descriptor);

        let duration = uniform_range(&mut rng, args.duration_min, args.duration_max);
        let t60 = uniform_range(&mut rng, args.t60_min, args.t60_max);
        let predelay_ms = rng.gen_range(0.0..=args.predelay_max_ms.max(0.0));

        descriptor.apply_overrides(Some(duration), Some(t60), Some(predelay_ms), None);
        descriptor.apply_spectral_overrides(
            maybe_jittered_unit(&mut rng, args.jitter, 0.55),
            None,
            None,
            None,
        );
        descriptor.apply_structure_overrides(
            maybe_jittered_unit(&mut rng, args.jitter, 0.45),
            maybe_jittered_unit(&mut rng, args.jitter, 0.7),
            maybe_jittered_unit(&mut rng, args.jitter, 0.65),
        );
        descriptor.apply_spatial_overrides(
            Some(channel_format_from_arg(args.channels)),
            maybe_jittered_unit(&mut rng, args.jitter, 0.75),
            maybe_jittered_unit(&mut rng, args.jitter, 0.55),
            None,
        );
        descriptor.clamp();

        let mut warnings = Vec::new();
        apply_duration_floor(&mut descriptor, false, &mut warnings);

        let mut generated = match generator.generate(&descriptor, sample_seed) {
            Ok(generated) => generated,
            Err(err) => {
                failures.push(format!("{id}: generation failed: {err}"));
                if failures.len() > args.max_failures {
                    break;
                }
                continue;
            }
        };
        if let Some(tail_fade_ms) = args.tail_fade_ms {
            if let Some(w) = util::audio::apply_tail_fade(
                &mut generated.channels,
                args.sample_rate,
                tail_fade_ms,
            ) {
                warnings.push(w);
            }
        }

        let ir_path = ir_dir.join(format!("{id}.wav"));
        if let Err(err) =
            util::audio::write_wav_f32(&ir_path, args.sample_rate, &generated.channels)
        {
            failures.push(format!("{id}: write failed: {err}"));
            if failures.len() > args.max_failures {
                break;
            }
            continue;
        }

        let channel_map = spatial::build_channel_map(&descriptor.spatial);
        if let Err(err) = spatial::validate_channel_map(&channel_map, generated.channels.len()) {
            failures.push(format!("{id}: channel map invalid: {err}"));
            if failures.len() > args.max_failures {
                break;
            }
            continue;
        }
        let channel_map_path = channel_map_dir.join(format!("{id}.channels.json"));
        util::json::write_pretty_json(&channel_map_path, &channel_map)?;

        let mut analysis = analyzer.analyze_with_channel_map(
            &generated.channels,
            args.sample_rate,
            Some(&channel_map),
        );
        analysis.warnings.extend(warnings);
        let quality = if args.quality_gate {
            Some(evaluate_quality_gate(&analysis, quality_profile))
        } else {
            None
        };
        if let Some(q) = quality.as_ref() {
            if !q.passed {
                analysis.warnings.push(format!(
                    "quality gate '{}' failed ({} checks)",
                    format!("{:?}", q.profile).to_lowercase(),
                    q.failed_checks.len()
                ));
            }
        }

        let metadata_path = metadata_dir.join(format!("{id}.json"));
        let analysis_path = analysis_dir.join(format!("{id}.json"));
        util::json::write_pretty_json(&analysis_path, &analysis)?;

        let metadata = GenerationMetadata {
            schema_version: "latent-ir.generation.v1".to_string(),
            project: "latent-ir".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            command: "dataset synth".to_string(),
            replay_command: build_replay_command(
                &prompt,
                preset.as_deref(),
                &descriptor,
                sample_seed,
                args.sample_rate,
                args.channels,
                args.tail_fade_ms,
            ),
            seed: sample_seed,
            prompt: Some(prompt.clone()),
            preset: preset.clone(),
            conditioning: ConditioningTrace::default(),
            sample_rate: args.sample_rate,
            spatial_encoding: channel_map.spatial_encoding.clone(),
            channel_format: channel_map.layout_name.clone(),
            channel_labels: descriptor.spatial.resolved_channel_labels(),
            channel_map_path: Some(rel(&channel_map_path, &args.out_dir)),
            ir_sha256: util::hash::sha256_channels_f32(&generated.channels),
            descriptor_sha256: util::hash::sha256_json(&descriptor)?,
            channel_map_sha256: util::hash::sha256_json(&channel_map)?,
            descriptor: descriptor.clone(),
            quality_gate_profile: quality
                .as_ref()
                .map(|q| format!("{:?}", q.profile).to_lowercase()),
            quality_gate_passed: quality.as_ref().map(|q| q.passed),
            quality_gate_failed_checks: quality.as_ref().map(|q| q.failed_checks.clone()),
            warnings: analysis.warnings.clone(),
            generated_at_utc: Utc::now(),
            analysis: analysis.clone(),
        };
        util::json::write_pretty_json(&metadata_path, &metadata)?;

        let record = DatasetRecord {
            id: id.clone(),
            seed: sample_seed,
            prompt: prompt.clone(),
            preset: preset.clone(),
            ir_wav: rel(&ir_path, &args.out_dir),
            metadata_json: rel(&metadata_path, &args.out_dir),
            analysis_json: rel(&analysis_path, &args.out_dir),
            channel_map_json: Some(rel(&channel_map_path, &args.out_dir)),
            descriptor: descriptor.clone(),
            duration_s: analysis.duration_s,
            t60_s_est: analysis.t60_s_est,
            predelay_ms_est: analysis.predelay_ms_est,
            decay_db_span: analysis.decay_db_span,
            quality_gate_passed: quality.as_ref().map(|q| q.passed),
        };
        records.push(record);

        if args.export_training_json {
            text_train.push(TextTrainSample {
                prompt: prompt.clone(),
                descriptor: descriptor.clone(),
            });
            audio_train.push(AudioTrainSample {
                audio_path: rel(&ir_path, &args.out_dir),
                descriptor: descriptor.clone(),
            });
        }
    }

    if records.is_empty() {
        anyhow::bail!(
            "dataset synthesis produced 0 successful samples (failures={})",
            failures.len()
        );
    }

    if !failures.is_empty() {
        let failures_path = args.out_dir.join("failures.log");
        let body = failures.join("\n");
        std::fs::write(&failures_path, body)?;
    }

    let summary = summarize(&records);
    let manifest = DatasetManifest {
        schema_version: "latent-ir.dataset.v1".to_string(),
        project: "latent-ir".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at_utc: Utc::now(),
        config: DatasetConfigSnapshot {
            count_requested: args.count,
            count_succeeded: records.len(),
            count_failed: failures.len(),
            seed: args.seed,
            sample_rate: args.sample_rate,
            channel_format: channel_arg_value(args.channels).to_string(),
            duration_range_s: [args.duration_min, args.duration_max],
            t60_range_s: [args.t60_min, args.t60_max],
            predelay_max_ms: args.predelay_max_ms,
            jitter: args.jitter,
            preset_mix: args.preset_mix,
            prompt_bank_size: prompt_bank.len(),
            quality_gate: args.quality_gate,
            quality_profile: if args.quality_gate {
                Some(format!("{:?}", quality_profile).to_lowercase())
            } else {
                None
            },
            tail_fade_ms: args.tail_fade_ms,
            export_training_json: args.export_training_json,
        },
        summary,
        records,
    };

    let manifest_path = args.out_dir.join("manifest.dataset.json");
    util::json::write_pretty_json(&manifest_path, &manifest)?;

    if args.export_training_json {
        util::json::write_pretty_json(args.out_dir.join("training_text.json"), &text_train)?;
        util::json::write_pretty_json(args.out_dir.join("training_audio.json"), &audio_train)?;
    }

    println!(
        "{}",
        util::console::info("dataset_manifest", manifest_path.display().to_string())
    );
    println!("{}", util::console::metric("samples_requested", args.count));
    println!(
        "{}",
        util::console::metric("samples_succeeded", manifest.config.count_succeeded)
    );
    println!(
        "{}",
        util::console::metric("samples_failed", manifest.config.count_failed)
    );
    println!(
        "{}",
        util::console::metric(
            "mean_t60_s_est",
            format!("{:.3}", manifest.summary.mean_t60_s_est)
        )
    );
    println!(
        "{}",
        util::console::metric(
            "quality_gate_failures",
            manifest.summary.quality_gate_failures
        )
    );
    Ok(())
}

fn run_split(args: DatasetSplitArgs) -> Result<()> {
    validate_split_args(&args)?;
    let text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("failed to read {}", args.manifest.display()))?;
    let manifest: DatasetManifest =
        serde_json::from_str(&text).with_context(|| "failed to parse dataset manifest JSON")?;

    anyhow::ensure!(
        !manifest.records.is_empty(),
        "dataset manifest has no records to split"
    );
    let source_manifest_sha256 = util::hash::sha256_hex(text.as_bytes());
    let root = args.manifest.parent().unwrap_or_else(|| Path::new("."));

    let mut records = manifest.records.clone();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    records.shuffle(&mut rng);

    let total = records.len();
    let train_n = ((total as f32) * args.train_ratio).round() as usize;
    let val_n = ((total as f32) * args.val_ratio).round() as usize;
    let train_n = train_n.min(total);
    let val_n = val_n.min(total.saturating_sub(train_n));
    let test_n = total.saturating_sub(train_n + val_n);

    let mut train = Vec::with_capacity(train_n);
    let mut val = Vec::with_capacity(val_n);
    let mut test = Vec::with_capacity(test_n);

    for (idx, rec) in records.iter().enumerate() {
        let split_rec = if args.lock_hashes {
            let meta_path = root.join(&rec.metadata_json);
            let meta_text = std::fs::read_to_string(&meta_path)
                .with_context(|| format!("failed to read {}", meta_path.display()))?;
            let meta: GenerationMetadata = serde_json::from_str(&meta_text)
                .with_context(|| format!("failed to parse {}", meta_path.display()))?;
            DatasetSplitRecord {
                id: rec.id.clone(),
                prompt: rec.prompt.clone(),
                ir_wav: rec.ir_wav.clone(),
                metadata_json: rec.metadata_json.clone(),
                analysis_json: rec.analysis_json.clone(),
                descriptor: rec.descriptor.clone(),
                ir_sha256: Some(meta.ir_sha256),
                descriptor_sha256: Some(meta.descriptor_sha256),
                channel_map_sha256: Some(meta.channel_map_sha256),
            }
        } else {
            DatasetSplitRecord {
                id: rec.id.clone(),
                prompt: rec.prompt.clone(),
                ir_wav: rec.ir_wav.clone(),
                metadata_json: rec.metadata_json.clone(),
                analysis_json: rec.analysis_json.clone(),
                descriptor: rec.descriptor.clone(),
                ir_sha256: None,
                descriptor_sha256: None,
                channel_map_sha256: None,
            }
        };

        if idx < train_n {
            train.push(split_rec);
        } else if idx < train_n + val_n {
            val.push(split_rec);
        } else {
            test.push(split_rec);
        }
    }

    let split_manifest = DatasetSplitManifest {
        schema_version: "latent-ir.dataset-split.v1".to_string(),
        project: "latent-ir".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at_utc: Utc::now(),
        source_manifest_path: args.manifest.display().to_string(),
        source_manifest_sha256,
        split_seed: args.seed,
        ratios: DatasetSplitRatios {
            train: args.train_ratio,
            val: args.val_ratio,
            test: args.test_ratio,
        },
        counts: DatasetSplitCounts {
            total,
            train: train.len(),
            val: val.len(),
            test: test.len(),
        },
        hash_locked: args.lock_hashes,
        train: train.clone(),
        val: val.clone(),
        test: test.clone(),
    };
    util::json::write_pretty_json(&args.output, &split_manifest)?;

    if args.emit_training_json {
        let out_root = args.output.parent().unwrap_or_else(|| Path::new("."));
        write_split_training_json(
            out_root.join("train_text.json"),
            out_root.join("train_audio.json"),
            &train,
        )?;
        write_split_training_json(
            out_root.join("val_text.json"),
            out_root.join("val_audio.json"),
            &val,
        )?;
        write_split_training_json(
            out_root.join("test_text.json"),
            out_root.join("test_audio.json"),
            &test,
        )?;
    }

    println!(
        "{}",
        util::console::info("split_manifest", args.output.display().to_string())
    );
    println!("{}", util::console::metric("total", total));
    println!(
        "{}",
        util::console::metric("train", split_manifest.counts.train)
    );
    println!(
        "{}",
        util::console::metric("val", split_manifest.counts.val)
    );
    println!(
        "{}",
        util::console::metric("test", split_manifest.counts.test)
    );
    println!(
        "{}",
        util::console::metric("hash_locked", split_manifest.hash_locked)
    );
    Ok(())
}

fn write_split_training_json(
    text_path: impl AsRef<Path>,
    audio_path: impl AsRef<Path>,
    split: &[DatasetSplitRecord],
) -> Result<()> {
    let text: Vec<TextTrainSample> = split
        .iter()
        .map(|r| TextTrainSample {
            prompt: r.prompt.clone(),
            descriptor: r.descriptor.clone(),
        })
        .collect();
    let audio: Vec<AudioTrainSample> = split
        .iter()
        .map(|r| AudioTrainSample {
            audio_path: r.ir_wav.clone(),
            descriptor: r.descriptor.clone(),
        })
        .collect();
    util::json::write_pretty_json(text_path, &text)?;
    util::json::write_pretty_json(audio_path, &audio)?;
    Ok(())
}

fn validate_split_args(args: &DatasetSplitArgs) -> Result<()> {
    anyhow::ensure!(args.train_ratio.is_finite(), "train-ratio must be finite");
    anyhow::ensure!(args.val_ratio.is_finite(), "val-ratio must be finite");
    anyhow::ensure!(args.test_ratio.is_finite(), "test-ratio must be finite");
    anyhow::ensure!(
        args.train_ratio >= 0.0 && args.val_ratio >= 0.0 && args.test_ratio >= 0.0,
        "split ratios must be >= 0"
    );
    let sum = args.train_ratio + args.val_ratio + args.test_ratio;
    anyhow::ensure!((sum - 1.0).abs() <= 1e-3, "split ratios must sum to 1.0");
    Ok(())
}

fn validate_args(args: &DatasetSynthesizeArgs) -> Result<()> {
    anyhow::ensure!(args.count > 0, "count must be > 0");
    anyhow::ensure!(
        (8_000..=768_000).contains(&args.sample_rate),
        "sample rate {} out of supported range [8000, 768000] Hz",
        args.sample_rate
    );
    ensure_finite_positive("duration-min", args.duration_min)?;
    ensure_finite_positive("duration-max", args.duration_max)?;
    anyhow::ensure!(
        args.duration_max >= args.duration_min,
        "duration-max must be >= duration-min"
    );
    ensure_finite_positive("t60-min", args.t60_min)?;
    ensure_finite_positive("t60-max", args.t60_max)?;
    anyhow::ensure!(args.t60_max >= args.t60_min, "t60-max must be >= t60-min");
    anyhow::ensure!(
        args.predelay_max_ms.is_finite() && args.predelay_max_ms >= 0.0,
        "predelay-max-ms must be finite and >= 0"
    );
    anyhow::ensure!(
        args.jitter.is_finite() && (0.0..=1.0).contains(&args.jitter),
        "jitter must be in [0,1]"
    );
    anyhow::ensure!(
        args.preset_mix.is_finite() && (0.0..=1.0).contains(&args.preset_mix),
        "preset-mix must be in [0,1]"
    );
    if let Some(ms) = args.tail_fade_ms {
        anyhow::ensure!(ms.is_finite() && ms >= 0.0, "tail-fade-ms must be >= 0");
    }
    Ok(())
}

fn ensure_finite_positive(name: &str, v: f32) -> Result<()> {
    anyhow::ensure!(v.is_finite() && v > 0.0, "{name} must be finite and > 0");
    Ok(())
}

fn load_prompt_bank(path: Option<&Path>) -> Result<Vec<String>> {
    if let Some(path) = path {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt bank {}", path.display()))?;
        let prompts: Vec<String> =
            serde_json::from_str(&text).with_context(|| "prompt bank must be JSON string array")?;
        let filtered: Vec<String> = prompts
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        anyhow::ensure!(!filtered.is_empty(), "prompt bank is empty after filtering");
        return Ok(filtered);
    }
    Ok(default_prompt_bank())
}

fn default_prompt_bank() -> Vec<String> {
    vec![
        "massive concrete grain silo with long metallic ring".to_string(),
        "abandoned rail tunnel dark wet walls".to_string(),
        "frozen cave with crystalline high-frequency shimmer".to_string(),
        "intimate wood chapel warm early reflections".to_string(),
        "steel bunker with harsh specular reflections".to_string(),
        "glass corridor with bright flutter".to_string(),
        "cathedral nave with distant source".to_string(),
        "industrial aircraft hangar broad stereo image".to_string(),
        "deep cistern with lingering low-end bloom".to_string(),
        "narrow service shaft metallic resonance".to_string(),
        "stone hall with low predelay and dense tail".to_string(),
        "dark impossible tunnel with infinite decay illusion".to_string(),
    ]
}

fn summarize(records: &[DatasetRecord]) -> DatasetSummary {
    let n = records.len().max(1) as f32;
    let mean_duration_s = records.iter().map(|r| r.duration_s).sum::<f32>() / n;
    let mean_t60_s_est = records
        .iter()
        .map(|r| r.t60_s_est.unwrap_or(0.0))
        .sum::<f32>()
        / n;
    let mean_predelay_ms_est = records.iter().map(|r| r.predelay_ms_est).sum::<f32>() / n;
    let mean_decay_db_span = records.iter().map(|r| r.decay_db_span).sum::<f32>() / n;
    let quality_gate_failures = records
        .iter()
        .filter(|r| r.quality_gate_passed == Some(false))
        .count();
    DatasetSummary {
        mean_duration_s,
        mean_t60_s_est,
        mean_predelay_ms_est,
        mean_decay_db_span,
        quality_gate_failures,
    }
}

fn maybe_jittered_unit(rng: &mut ChaCha8Rng, jitter: f32, center: f32) -> Option<f32> {
    if rng.gen_bool(0.7) {
        Some((center + rng.gen_range(-jitter..=jitter)).clamp(0.0, 1.0))
    } else {
        None
    }
}

fn uniform_range(rng: &mut ChaCha8Rng, min: f32, max: f32) -> f32 {
    if (max - min).abs() <= f32::EPSILON {
        min
    } else {
        rng.gen_range(min..=max)
    }
}

fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
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

    let original = descriptor.time.duration;
    descriptor.time.duration = recommended;
    warnings.push(format!(
        "duration auto-extended from {:.3}s to {:.3}s to better capture requested decay",
        original, recommended
    ));
}

fn build_replay_command(
    prompt: &str,
    preset: Option<&str>,
    descriptor: &DescriptorSet,
    seed: u64,
    sample_rate: u32,
    channels: ChannelFormatArg,
    tail_fade_ms: Option<f32>,
) -> String {
    let mut parts = vec![
        "latent-ir".to_string(),
        "generate".to_string(),
        "--prompt".to_string(),
        shell_quote(prompt),
    ];
    if let Some(preset) = preset {
        parts.push("--preset".to_string());
        parts.push(shell_quote(preset));
    }
    parts.push("--duration".to_string());
    parts.push(format!("{:.6}", descriptor.time.duration));
    parts.push("--t60".to_string());
    parts.push(format!("{:.6}", descriptor.time.t60));
    parts.push("--predelay-ms".to_string());
    parts.push(format!("{:.6}", descriptor.time.predelay_ms));
    parts.push("--sample-rate".to_string());
    parts.push(sample_rate.to_string());
    parts.push("--seed".to_string());
    parts.push(seed.to_string());
    parts.push("--channels".to_string());
    parts.push(channel_arg_value(channels).to_string());
    if let Some(ms) = tail_fade_ms {
        parts.push("--tail-fade-ms".to_string());
        parts.push(format!("{ms}"));
    }
    parts.join(" ")
}

fn shell_quote(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/')
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn quality_profile_from_arg(arg: QualityProfileArg) -> QualityProfile {
    match arg {
        QualityProfileArg::Lenient => QualityProfile::Lenient,
        QualityProfileArg::Launch => QualityProfile::Launch,
        QualityProfileArg::Strict => QualityProfile::Strict,
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
