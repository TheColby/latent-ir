use anyhow::{Context, Result};

use crate::cli::MorphArgs;
use crate::core::morph::IrMorpher;
use crate::core::util;

pub fn run(args: MorphArgs) -> Result<()> {
    let alpha = args.alpha.clamp(0.0, 1.0);
    let a = util::audio::read_wav_f32(&args.ir_a)
        .with_context(|| format!("failed to read {}", args.ir_a.display()))?;
    let b = util::audio::read_wav_f32(&args.ir_b)
        .with_context(|| format!("failed to read {}", args.ir_b.display()))?;

    anyhow::ensure!(a.sample_rate == b.sample_rate, "sample rates must match");

    let out = IrMorpher::default().morph(&a.channels, &b.channels, alpha);
    util::audio::write_wav_f32(&args.output, a.sample_rate, &out)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    println!("wrote morphed IR: {}", args.output.display());
    Ok(())
}
