use anyhow::{Context, Result};

use crate::cli::{RenderArgs, RenderEngineArg};
use crate::core::render::{RenderEngine, RenderOptions, Renderer};
use crate::core::util;

pub fn run(args: RenderArgs) -> Result<()> {
    let mix = args.mix.clamp(0.0, 1.0);
    let input = util::audio::read_wav_f32(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let ir = util::audio::read_wav_f32(&args.ir)
        .with_context(|| format!("failed to read {}", args.ir.display()))?;

    anyhow::ensure!(
        input.sample_rate == ir.sample_rate,
        "sample rates must match"
    );

    let engine = match args.engine {
        RenderEngineArg::Auto => RenderEngine::Auto,
        RenderEngineArg::Direct => RenderEngine::Direct,
        RenderEngineArg::FftPartitioned => RenderEngine::FftPartitioned,
    };
    let options = RenderOptions {
        engine,
        partition_size: args.partition_size,
    };
    let rendered = Renderer::default().render_convolution_with_options(
        &input.channels,
        &ir.channels,
        mix,
        options,
    );
    util::audio::write_wav_f32(&args.output, input.sample_rate, &rendered)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    println!("wrote rendered audio: {}", args.output.display());
    Ok(())
}
