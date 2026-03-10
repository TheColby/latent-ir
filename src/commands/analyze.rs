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
        println!("file: {}", args.input.display());
        println!("sample_rate: {}", report.sample_rate);
        println!("duration_s: {:.3}", report.duration_s);
        println!("peak: {:.4}", report.peak);
        println!("rms: {:.4}", report.rms);
        println!("predelay_ms_est: {:.2}", report.predelay_ms_est);
        println!("edt_s_est: {:.3}", report.edt_s_est.unwrap_or(-1.0));
        println!("t20_s_est: {:.3}", report.t20_s_est.unwrap_or(-1.0));
        println!("t30_s_est: {:.3}", report.t30_s_est.unwrap_or(-1.0));
        println!("t60_s_est: {:.3}", report.t60_s_est.unwrap_or(-1.0));
        println!("spectral_centroid_hz: {:.1}", report.spectral_centroid_hz);
    }

    Ok(())
}
