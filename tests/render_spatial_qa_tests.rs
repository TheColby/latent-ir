use approx::assert_relative_eq;
use latent_ir::core::render::{RenderEngine, RenderOptions, Renderer};

#[test]
fn streaming_render_matches_direct_for_multichannel_beds() {
    let channels = 13usize; // 7.2.4 bed
    let input_len = 3072usize;
    let ir_len = 1536usize;
    let partition_size = 256usize;

    let input = synth_signal(channels, input_len);
    let ir = synth_ir(channels, ir_len);

    let direct = Renderer.render_convolution_with_options(
        &input,
        &ir,
        1.0,
        RenderOptions {
            engine: RenderEngine::Direct,
            partition_size,
        },
    );
    let mut streamed = collect_streaming(&input, &ir, partition_size);
    normalize_like_renderer(&mut streamed);

    assert_eq!(direct.len(), channels);
    assert_eq!(streamed.len(), channels);
    let expected_len = input_len + ir_len - 1;
    for ch in 0..channels {
        assert_eq!(direct[ch].len(), expected_len);
        assert_eq!(streamed[ch].len(), expected_len);
    }

    for ch in 0..channels {
        for (a, b) in direct[ch].iter().zip(streamed[ch].iter()) {
            assert_relative_eq!(a, b, epsilon = 4e-4);
        }
    }
}

#[test]
fn streaming_render_has_no_block_boundary_artifacts_for_custom_array() {
    let channels = 16usize;
    let input_len = 4096usize;
    let ir_len = 2048usize;
    let partition_size = 256usize;

    let input = synth_signal(channels, input_len);
    let ir = synth_ir(channels, ir_len);

    let direct = Renderer.render_convolution_with_options(
        &input,
        &ir,
        0.85,
        RenderOptions {
            engine: RenderEngine::Direct,
            partition_size,
        },
    );
    let mut streamed = collect_streaming_with_mix(&input, &ir, partition_size, 0.85);
    normalize_like_renderer(&mut streamed);

    let expected_len = input_len + ir_len - 1;
    for ch in 0..channels {
        assert_eq!(direct[ch].len(), expected_len);
        assert_eq!(streamed[ch].len(), expected_len);
    }

    let mut max_err = 0.0f32;
    for ch in 0..channels {
        for i in 0..expected_len {
            max_err = max_err.max((direct[ch][i] - streamed[ch][i]).abs());
        }
    }
    assert!(max_err < 8e-4, "max sample error too high: {max_err:.6}");

    for boundary in (partition_size..expected_len).step_by(partition_size) {
        let start = boundary.saturating_sub(2);
        let end = (boundary + 2).min(expected_len - 1);
        for ch in 0..channels {
            for i in start..=end {
                let err = (direct[ch][i] - streamed[ch][i]).abs();
                assert!(
                    err < 8e-4,
                    "boundary artifact at ch={} idx={} err={:.6}",
                    ch,
                    i,
                    err
                );
            }
        }
    }
}

fn synth_signal(channels: usize, len: usize) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; len]; channels];
    for (ch, row) in out.iter_mut().enumerate() {
        for (i, s) in row.iter_mut().enumerate() {
            let t = i as f32;
            *s = (t * (0.007 + ch as f32 * 0.0002)).sin() * 0.25
                + (t * (0.013 + ch as f32 * 0.0003)).cos() * 0.12;
        }
    }
    out
}

fn synth_ir(channels: usize, len: usize) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; len]; channels];
    for (ch, row) in out.iter_mut().enumerate() {
        row[0] = 1.0;
        for i in 1..len {
            let tau = 260.0 + (ch as f32 * 19.0);
            let decay = (-(i as f32 / tau)).exp();
            let alt = if (i + ch) % 9 == 0 { -1.0 } else { 1.0 };
            row[i] = alt * decay * (0.02 + ch as f32 * 0.0005);
        }
    }
    out
}

fn collect_streaming(input: &[Vec<f32>], ir: &[Vec<f32>], partition_size: usize) -> Vec<Vec<f32>> {
    collect_streaming_with_mix(input, ir, partition_size, 1.0)
}

fn collect_streaming_with_mix(
    input: &[Vec<f32>],
    ir: &[Vec<f32>],
    partition_size: usize,
    mix: f32,
) -> Vec<Vec<f32>> {
    let channels = input.len().max(ir.len()).max(1);
    let mut streamed = vec![Vec::<f32>::new(); channels];
    Renderer.render_convolution_streaming_blocks(input, ir, mix, partition_size, |block, valid| {
        for ch in 0..channels {
            streamed[ch].extend_from_slice(&block[ch][..valid]);
        }
    });
    streamed
}

fn normalize_like_renderer(channels: &mut [Vec<f32>]) {
    let peak = channels
        .iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 1e-9 {
        return;
    }
    let gain = (0.98 / peak).min(1.0);
    for ch in channels {
        for s in ch {
            *s *= gain;
        }
    }
}
