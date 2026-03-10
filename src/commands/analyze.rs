use anyhow::{Context, Result};

use crate::cli::AnalyzeArgs;
use crate::core::analysis::IrAnalyzer;
use crate::core::util;

pub fn run(args: AnalyzeArgs) -> Result<()> {
    let wav = util::audio::read_wav_f32(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;

    let report = IrAnalyzer::default().analyze(&wav.channels, wav.sample_rate);

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
            util::console::metric("sample_rate", report.sample_rate)
        );
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
    }

    Ok(())
}
