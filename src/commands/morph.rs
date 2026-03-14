use anyhow::{Context, Result};

use crate::cli::MorphArgs;
use crate::core::morph::IrMorpher;
use crate::core::util;

pub fn run(args: MorphArgs) -> Result<()> {
    let alpha = args.alpha.clamp(0.0, 1.0);
    let a = util::audio::read_wav_f32(&args.ir_a)
        .with_context(|| format!("failed to read {}", args.ir_a.display()))?;
    let mut b = util::audio::read_wav_f32(&args.ir_b)
        .with_context(|| format!("failed to read {}", args.ir_b.display()))?;

    if a.sample_rate != b.sample_rate {
        if args.auto_resample {
            println!(
                "{}",
                util::console::warning(&format!(
                    "auto-resampling second IR from {} Hz to {} Hz (linear)",
                    b.sample_rate, a.sample_rate
                ))
            );
            b.channels = util::audio::resample_linear(&b.channels, b.sample_rate, a.sample_rate);
            b.sample_rate = a.sample_rate;
        } else {
            anyhow::bail!(
                "sample rates must match (ir_a={} Hz, ir_b={} Hz). Re-run with --auto-resample to reconcile automatically",
                a.sample_rate,
                b.sample_rate
            );
        }
    }

    let out = IrMorpher::default().morph(&a.channels, &b.channels, alpha);
    util::audio::write_wav_f32(&args.output, a.sample_rate, &out)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    println!("wrote morphed IR: {}", args.output.display());
    Ok(())
}
