use rustfft::{num_complex::Complex32, FftPlanner};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderEngine {
    Auto,
    Direct,
    FftPartitioned,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    pub engine: RenderEngine,
    pub partition_size: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            engine: RenderEngine::Auto,
            partition_size: 2048,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Renderer;

impl Renderer {
    pub fn render_convolution(
        &self,
        input: &[Vec<f32>],
        ir: &[Vec<f32>],
        mix: f32,
    ) -> Vec<Vec<f32>> {
        self.render_convolution_with_options(input, ir, mix, RenderOptions::default())
    }

    pub fn render_convolution_with_options(
        &self,
        input: &[Vec<f32>],
        ir: &[Vec<f32>],
        mix: f32,
        options: RenderOptions,
    ) -> Vec<Vec<f32>> {
        let mix = mix.clamp(0.0, 1.0);
        let channels = input.len().max(ir.len()).max(1);

        let mut wet = Vec::with_capacity(channels);
        for ch in 0..channels {
            let in_ch = input
                .get(ch)
                .or_else(|| input.first())
                .cloned()
                .unwrap_or_default();
            let ir_ch = ir
                .get(ch)
                .or_else(|| ir.first())
                .cloned()
                .unwrap_or_else(|| vec![1.0]);

            let selected = match options.engine {
                RenderEngine::Direct => convolve_direct(&in_ch, &ir_ch),
                RenderEngine::FftPartitioned => {
                    convolve_fft_partitioned(&in_ch, &ir_ch, options.partition_size)
                }
                RenderEngine::Auto => {
                    if should_use_fft(in_ch.len(), ir_ch.len()) {
                        convolve_fft_partitioned(&in_ch, &ir_ch, options.partition_size)
                    } else {
                        convolve_direct(&in_ch, &ir_ch)
                    }
                }
            };

            wet.push(selected);
        }

        let out_len = wet.iter().map(Vec::len).max().unwrap_or(0);
        let mut out = vec![vec![0.0f32; out_len]; channels];

        for ch in 0..channels {
            let dry = input.get(ch).or_else(|| input.first());
            let wet_ch = &wet[ch];
            for i in 0..out_len {
                let d = dry.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                let w = wet_ch.get(i).copied().unwrap_or(0.0);
                out[ch][i] = d * (1.0 - mix) + w * mix;
            }
        }

        normalize(&mut out);
        out
    }
}

fn should_use_fft(input_len: usize, ir_len: usize) -> bool {
    input_len.saturating_mul(ir_len) > 2_000_000
}

fn convolve_direct(x: &[f32], h: &[f32]) -> Vec<f32> {
    if x.is_empty() || h.is_empty() {
        return vec![];
    }
    let mut y = vec![0.0f32; x.len() + h.len() - 1];
    for (i, &xs) in x.iter().enumerate() {
        for (j, &hs) in h.iter().enumerate() {
            y[i + j] += xs * hs;
        }
    }
    y
}

fn convolve_fft_partitioned(x: &[f32], h: &[f32], partition_size: usize) -> Vec<f32> {
    if x.is_empty() || h.is_empty() {
        return vec![];
    }

    let b = partition_size.max(64);
    let n_fft = (2 * b).next_power_of_two();
    let output_len = x.len() + h.len() - 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    let partitions = h.len().div_ceil(b);
    let mut h_parts_fft = Vec::with_capacity(partitions);
    for p in 0..partitions {
        let start = p * b;
        let end = (start + b).min(h.len());

        let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
        for (i, &v) in h[start..end].iter().enumerate() {
            buf[i].re = v;
        }
        fft.process(&mut buf);
        h_parts_fft.push(buf);
    }

    let input_blocks = x.len().div_ceil(b);
    let mut out = vec![0.0f32; output_len];

    for blk in 0..input_blocks {
        let start = blk * b;
        let end = (start + b).min(x.len());

        let mut x_fft = vec![Complex32::new(0.0, 0.0); n_fft];
        for (i, &v) in x[start..end].iter().enumerate() {
            x_fft[i].re = v;
        }
        fft.process(&mut x_fft);

        for (p, h_fft) in h_parts_fft.iter().enumerate() {
            let mut y_fft = vec![Complex32::new(0.0, 0.0); n_fft];
            for i in 0..n_fft {
                y_fft[i] = x_fft[i] * h_fft[i];
            }
            ifft.process(&mut y_fft);

            let time_offset = (blk + p) * b;
            for (i, c) in y_fft.iter().enumerate() {
                let out_idx = time_offset + i;
                if out_idx >= output_len {
                    break;
                }
                out[out_idx] += c.re / n_fft as f32;
            }
        }
    }

    out
}

fn normalize(channels: &mut [Vec<f32>]) {
    let peak = channels
        .iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f32, |m, &s| m.max(s.abs()));
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
