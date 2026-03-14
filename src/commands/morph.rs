use anyhow::{Context, Result};

use crate::cli::{MorphArgs, ResampleModeArg};
use crate::core::morph::{AlphaTrajectory, IrMorpher};
use crate::core::util;

pub fn run(args: MorphArgs) -> Result<()> {
    let alpha = if args.alpha_trajectory.is_none() {
        anyhow::ensure!(args.alpha.is_finite(), "alpha must be a finite number");
        let alpha = args.alpha.clamp(0.0, 1.0);
        if (alpha - args.alpha).abs() > f32::EPSILON {
            println!(
                "{}",
                util::console::warning(&format!(
                    "alpha {:.4} clamped to {:.4} (valid range [0,1])",
                    args.alpha, alpha
                ))
            );
        }
        Some(alpha)
    } else {
        None
    };

    let trajectory = if let Some(path) = args.alpha_trajectory.as_deref() {
        println!(
            "{}",
            util::console::info("alpha_trajectory", path.display().to_string())
        );
        Some(AlphaTrajectory::from_json_file(path)?)
    } else {
        None
    };

    if trajectory.is_some() {
        println!(
            "{}",
            util::console::warning(
                "--alpha-trajectory provided; static --alpha value is ignored in this run"
            )
        );
    }
    let a = util::audio::read_wav_f32(&args.ir_a)
        .with_context(|| format!("failed to read {}", args.ir_a.display()))?;
    let mut b = util::audio::read_wav_f32(&args.ir_b)
        .with_context(|| format!("failed to read {}", args.ir_b.display()))?;

    if a.sample_rate != b.sample_rate {
        if args.auto_resample {
            let mode = resample_mode_from_arg(args.resample_mode);
            println!(
                "{}",
                util::console::warning(&format!(
                    "auto-resampling second IR from {} Hz to {} Hz ({})",
                    b.sample_rate,
                    a.sample_rate,
                    resample_mode_name(mode)
                ))
            );
            b.channels = util::audio::resample(&b.channels, b.sample_rate, a.sample_rate, mode);
            b.sample_rate = a.sample_rate;
        } else {
            anyhow::bail!(
                "sample rates must match (ir_a={} Hz, ir_b={} Hz). Re-run with --auto-resample to reconcile automatically",
                a.sample_rate,
                b.sample_rate
            );
        }
    }

    let out = if let Some(traj) = trajectory.as_ref() {
        IrMorpher::default().morph_with_trajectory(&a.channels, &b.channels, traj)
    } else {
        IrMorpher::default().morph(&a.channels, &b.channels, alpha.unwrap_or(0.5))
    };
    util::audio::write_wav_f32(&args.output, a.sample_rate, &out)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    println!("wrote morphed IR: {}", args.output.display());
    Ok(())
}

fn resample_mode_from_arg(arg: ResampleModeArg) -> util::audio::ResampleMode {
    match arg {
        ResampleModeArg::Linear => util::audio::ResampleMode::Linear,
        ResampleModeArg::Cubic => util::audio::ResampleMode::Cubic,
    }
}

fn resample_mode_name(mode: util::audio::ResampleMode) -> &'static str {
    match mode {
        util::audio::ResampleMode::Linear => "linear",
        util::audio::ResampleMode::Cubic => "cubic",
    }
}
