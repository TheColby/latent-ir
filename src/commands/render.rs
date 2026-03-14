use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::cli::{RenderArgs, RenderEngineArg};
use crate::core::render::{RenderEngine, RenderOptions, Renderer};
use crate::core::util;

const AUTO_STREAMING_WORKLOAD_SAMPLES: usize = 40_000_000;
const AUTO_STREAMING_IR_LEN: usize = 262_144;

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

    let input_len = max_channel_len(&input.channels);
    let ir_len = max_channel_len(&ir.channels);
    let channel_count = input.channels.len().max(ir.channels.len()).max(1);

    let engine = resolve_engine(args.engine, input_len, ir_len, channel_count);
    println!(
        "{}",
        util::console::metric("render_engine", render_engine_name(engine))
    );
    if args.engine == RenderEngineArg::Auto {
        println!(
            "{}",
            util::console::info(
                "render_engine_auto_reason",
                format!(
                    "channels={channel_count}, input_len={input_len}, ir_len={ir_len}, workload_samples={}",
                    input_len
                        .saturating_add(ir_len)
                        .saturating_sub(1)
                        .saturating_mul(channel_count)
                )
            )
        );
    }

    let options = RenderOptions {
        engine,
        partition_size: args.partition_size,
    };
    if engine == RenderEngine::FftStreaming {
        write_streaming_render(
            &Renderer::default(),
            &input.channels,
            &ir.channels,
            mix,
            options.partition_size,
            input.sample_rate,
            &args.output,
        )
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    } else {
        let rendered = Renderer::default().render_convolution_with_options(
            &input.channels,
            &ir.channels,
            mix,
            options,
        );
        util::audio::write_wav_f32(&args.output, input.sample_rate, &rendered)
            .with_context(|| format!("failed to write {}", args.output.display()))?;
    }
    println!("wrote rendered audio: {}", args.output.display());
    Ok(())
}

fn max_channel_len(channels: &[Vec<f32>]) -> usize {
    channels.iter().map(Vec::len).max().unwrap_or(0)
}

fn resolve_engine(
    arg: RenderEngineArg,
    input_len: usize,
    ir_len: usize,
    channels: usize,
) -> RenderEngine {
    match arg {
        RenderEngineArg::Auto => auto_engine_for_sizes(input_len, ir_len, channels),
        RenderEngineArg::Direct => RenderEngine::Direct,
        RenderEngineArg::FftPartitioned => RenderEngine::FftPartitioned,
        RenderEngineArg::FftStreaming => RenderEngine::FftStreaming,
    }
}

fn auto_engine_for_sizes(input_len: usize, ir_len: usize, channels: usize) -> RenderEngine {
    let c = channels.max(1);
    let out_len = input_len.saturating_add(ir_len).saturating_sub(1);
    let workload = out_len.saturating_mul(c);
    if workload >= AUTO_STREAMING_WORKLOAD_SAMPLES || ir_len >= AUTO_STREAMING_IR_LEN {
        return RenderEngine::FftStreaming;
    }
    if input_len.saturating_mul(ir_len) > 2_000_000 {
        return RenderEngine::FftPartitioned;
    }
    RenderEngine::Direct
}

fn render_engine_name(engine: RenderEngine) -> &'static str {
    match engine {
        RenderEngine::Auto => "auto",
        RenderEngine::Direct => "direct",
        RenderEngine::FftPartitioned => "fft-partitioned",
        RenderEngine::FftStreaming => "fft-streaming",
    }
}

fn write_streaming_render(
    renderer: &Renderer,
    input: &[Vec<f32>],
    ir: &[Vec<f32>],
    mix: f32,
    partition_size: usize,
    sample_rate: u32,
    output_path: &std::path::Path,
) -> Result<()> {
    let channels = input.len().max(ir.len()).max(1);

    let mut peak = 0.0f32;
    renderer.render_convolution_streaming_blocks(input, ir, mix, partition_size, |block, valid| {
        for ch in block.iter().take(channels) {
            for &s in ch.iter().take(valid) {
                peak = peak.max(s.abs());
            }
        }
    });
    let norm = if peak <= 1e-9 {
        1.0
    } else {
        (0.98 / peak).min(1.0)
    };

    let spec = WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(output_path, spec)?;
    let mut write_error: Option<anyhow::Error> = None;

    renderer.render_convolution_streaming_blocks(input, ir, mix, partition_size, |block, valid| {
        if write_error.is_some() {
            return;
        }
        for i in 0..valid {
            for ch in block.iter().take(channels) {
                let s = (ch[i] * norm).clamp(-1.0, 1.0);
                if let Err(err) = writer.write_sample(s) {
                    write_error = Some(err.into());
                    return;
                }
            }
        }
    });

    if let Some(err) = write_error {
        return Err(err);
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{auto_engine_for_sizes, RenderEngine};

    #[test]
    fn auto_prefers_direct_for_small_cases() {
        assert_eq!(auto_engine_for_sizes(2_000, 512, 2), RenderEngine::Direct);
    }

    #[test]
    fn auto_prefers_fft_partitioned_for_mid_cases() {
        assert_eq!(
            auto_engine_for_sizes(240_000, 48_000, 2),
            RenderEngine::FftPartitioned
        );
    }

    #[test]
    fn auto_prefers_streaming_for_large_workloads() {
        assert_eq!(
            auto_engine_for_sizes(10_000_000, 200_000, 12),
            RenderEngine::FftStreaming
        );
        assert_eq!(
            auto_engine_for_sizes(100_000, 300_000, 2),
            RenderEngine::FftStreaming
        );
    }
}
