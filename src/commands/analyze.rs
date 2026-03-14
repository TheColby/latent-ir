use anyhow::{Context, Result};

use crate::cli::{AnalyzeArgs, QualityProfileArg};
use crate::core::analysis::{evaluate_quality_gate, IrAnalyzer, QualityProfile};
use crate::core::spatial;
use crate::core::util;

pub fn run(args: AnalyzeArgs) -> Result<()> {
    let wav = util::audio::read_wav_f32(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;

    let channel_map = if let Some(path) = args.channel_map.as_deref() {
        Some(spatial::read_channel_map(path)?)
    } else {
        spatial::try_read_companion_channel_map(&args.input)?
    };

    let report = IrAnalyzer::default().analyze_with_channel_map(
        &wav.channels,
        wav.sample_rate,
        channel_map.as_ref(),
    );

    if args.json {
        if let Some(path) = args.output {
            util::json::write_pretty_json(path, &report)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    } else {
        println!(
            "{}",
            util::console::info("file", args.input.display().to_string())
        );
        println!(
            "{}",
            util::console::warning(
                "metrics_note: engineering estimates for v0 (not standards-certified architectural acoustics metrology)"
            )
        );
        println!(
            "{}",
            util::console::metric("sample_rate", report.sample_rate)
        );
        if let Some(map) = channel_map.as_ref() {
            println!(
                "{}",
                util::console::metric("channel_format", &map.layout_name)
            );
            println!(
                "{}",
                util::console::metric("spatial_encoding", &map.spatial_encoding)
            );
            let labels = map
                .channels
                .iter()
                .map(|c| c.label.as_str())
                .collect::<Vec<_>>()
                .join(",");
            println!("{}", util::console::metric("channel_labels", labels));
        }
        println!(
            "{}",
            util::console::metric("duration_s", format!("{:.3}", report.duration_s))
        );
        println!(
            "{}",
            util::console::metric("peak", format!("{:.4}", report.peak))
        );
        println!(
            "{}",
            util::console::metric("rms", format!("{:.4}", report.rms))
        );
        println!(
            "{}",
            util::console::metric("predelay_ms_est", format!("{:.2}", report.predelay_ms_est))
        );
        println!(
            "{}",
            util::console::metric(
                "edt_s_est",
                format!("{:.3}", report.edt_s_est.unwrap_or(-1.0))
            )
        );
        println!(
            "{}",
            util::console::metric(
                "t20_s_est",
                format!("{:.3}", report.t20_s_est.unwrap_or(-1.0))
            )
        );
        println!(
            "{}",
            util::console::metric(
                "t30_s_est",
                format!("{:.3}", report.t30_s_est.unwrap_or(-1.0))
            )
        );
        println!(
            "{}",
            util::console::metric(
                "t60_s_est",
                format!("{:.3}", report.t60_s_est.unwrap_or(-1.0))
            )
        );
        println!(
            "{}",
            util::console::metric("decay_db_span", format!("{:.2}", report.decay_db_span))
        );
        println!(
            "{}",
            util::console::metric_opt("t60_confidence", fmt_opt(report.t60_confidence, 3))
        );
        println!(
            "{}",
            util::console::metric_opt("edt_confidence", fmt_opt(report.edt_confidence, 3))
        );
        println!(
            "{}",
            util::console::metric("crest_factor_db", format!("{:.2}", report.crest_factor_db))
        );
        println!(
            "{}",
            util::console::metric_opt(
                "tail_reaches_minus60db_s",
                fmt_opt(report.tail_reaches_minus60db_s, 3)
            )
        );
        println!(
            "{}",
            util::console::metric_opt(
                "tail_margin_to_end_s",
                fmt_opt(report.tail_margin_to_end_s, 3)
            )
        );
        println!(
            "{}",
            util::console::metric(
                "spectral_centroid_hz",
                format!("{:.1}", report.spectral_centroid_hz)
            )
        );
        println!(
            "{}",
            util::console::metric_opt(
                "inter_channel_corr_mean_abs",
                fmt_opt(report.inter_channel_correlation_mean_abs, 4)
            )
        );
        println!(
            "{}",
            util::console::metric_opt(
                "inter_channel_corr_min_abs",
                fmt_opt(report.inter_channel_correlation_min_abs, 4)
            )
        );
        println!(
            "{}",
            util::console::metric_opt("arrival_min_ms", fmt_opt(report.arrival_min_ms, 3))
        );
        println!(
            "{}",
            util::console::metric_opt("arrival_max_ms", fmt_opt(report.arrival_max_ms, 3))
        );
        println!(
            "{}",
            util::console::metric_opt("arrival_spread_ms", fmt_opt(report.arrival_spread_ms, 3))
        );
        println!(
            "{}",
            util::console::metric_opt("itd_01_ms", fmt_opt(report.itd_01_ms, 3))
        );
        println!(
            "{}",
            util::console::metric_opt("iacc_early_01", fmt_opt(report.iacc_early_01, 5))
        );
        println!(
            "{}",
            util::console::metric_opt(
                "inter_channel_itd_mean_abs_ms",
                fmt_opt(report.inter_channel_itd_mean_abs_ms, 3)
            )
        );
        println!(
            "{}",
            util::console::metric_opt(
                "inter_channel_itd_max_abs_ms",
                fmt_opt(report.inter_channel_itd_max_abs_ms, 3)
            )
        );
        println!(
            "{}",
            util::console::metric_opt(
                "inter_channel_iacc_early_mean",
                fmt_opt(report.inter_channel_iacc_early_mean, 5)
            )
        );
        println!(
            "{}",
            util::console::metric_opt(
                "inter_channel_iacc_early_min",
                fmt_opt(report.inter_channel_iacc_early_min, 5)
            )
        );
        println!(
            "{}",
            util::console::metric_opt("front_energy_ratio", fmt_opt(report.front_energy_ratio, 4))
        );
        println!(
            "{}",
            util::console::metric_opt("rear_energy_ratio", fmt_opt(report.rear_energy_ratio, 4))
        );
        println!(
            "{}",
            util::console::metric_opt(
                "height_energy_ratio",
                fmt_opt(report.height_energy_ratio, 4)
            )
        );
        println!(
            "{}",
            util::console::metric_opt("lfe_energy_ratio", fmt_opt(report.lfe_energy_ratio, 4))
        );
        if !report.warnings.is_empty() {
            println!("{}", util::console::warning("warnings:"));
            for w in &report.warnings {
                println!("  {}", util::console::warning(&format!("- {w}")));
            }
        }
    }

    if args.quality_gate {
        let gate = evaluate_quality_gate(&report, quality_profile_from_arg(args.quality_profile));
        if !args.json {
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
        if !gate.passed {
            anyhow::bail!(
                "quality gate failed for profile '{}' ({} checks)",
                format!("{:?}", gate.profile).to_lowercase(),
                gate.failed_checks.len()
            );
        }
    }

    Ok(())
}

fn fmt_opt(value: Option<f32>, decimals: usize) -> Option<String> {
    value.map(|v| format!("{:.*}", decimals, v))
}

fn quality_profile_from_arg(arg: QualityProfileArg) -> QualityProfile {
    match arg {
        QualityProfileArg::Lenient => QualityProfile::Lenient,
        QualityProfileArg::Launch => QualityProfile::Launch,
        QualityProfileArg::Strict => QualityProfile::Strict,
    }
}
