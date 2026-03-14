use anyhow::{Context, Result};

use crate::cli::AnalyzeArgs;
use crate::core::analysis::IrAnalyzer;
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
            util::console::metric(
                "spectral_centroid_hz",
                format!("{:.1}", report.spectral_centroid_hz)
            )
        );
        println!(
            "{}",
            util::console::metric(
                "inter_channel_corr_mean_abs",
                report
                    .inter_channel_correlation_mean_abs
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "inter_channel_corr_min_abs",
                report
                    .inter_channel_correlation_min_abs
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "arrival_min_ms",
                report
                    .arrival_min_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "arrival_max_ms",
                report
                    .arrival_max_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "arrival_spread_ms",
                report
                    .arrival_spread_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "itd_01_ms",
                report
                    .itd_01_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "iacc_early_01",
                report
                    .iacc_early_01
                    .map(|v| format!("{v:.5}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "inter_channel_itd_mean_abs_ms",
                report
                    .inter_channel_itd_mean_abs_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "inter_channel_itd_max_abs_ms",
                report
                    .inter_channel_itd_max_abs_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "inter_channel_iacc_early_mean",
                report
                    .inter_channel_iacc_early_mean
                    .map(|v| format!("{v:.5}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "inter_channel_iacc_early_min",
                report
                    .inter_channel_iacc_early_min
                    .map(|v| format!("{v:.5}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "front_energy_ratio",
                report
                    .front_energy_ratio
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "rear_energy_ratio",
                report
                    .rear_energy_ratio
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "height_energy_ratio",
                report
                    .height_energy_ratio
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
        println!(
            "{}",
            util::console::metric(
                "lfe_energy_ratio",
                report
                    .lfe_energy_ratio
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "n/a".to_string())
            )
        );
    }

    Ok(())
}
