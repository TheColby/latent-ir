use rustfft::{num_complex::Complex32, FftPlanner};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderEngine {
    Auto,
    Direct,
    FftPartitioned,
    FftStreaming,
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
                RenderEngine::FftStreaming => {
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

    /// Streaming block-wise convolution path intended for long-form renders.
    /// The callback receives one output block and the valid frame count in that block.
    pub fn render_convolution_streaming_blocks<F>(
        &self,
        input: &[Vec<f32>],
        ir: &[Vec<f32>],
        mix: f32,
        partition_size: usize,
        mut on_block: F,
    ) where
        F: FnMut(&[Vec<f32>], usize),
    {
        let mix = mix.clamp(0.0, 1.0);
        let channels = input.len().max(ir.len()).max(1);
        let in_len = input
            .iter()
            .map(Vec::len)
            .max()
            .or_else(|| input.first().map(Vec::len))
            .unwrap_or(0);
        let ir_len = ir
            .iter()
            .map(Vec::len)
            .max()
            .or_else(|| ir.first().map(Vec::len))
            .unwrap_or(1)
            .max(1);
        let out_len = in_len.saturating_add(ir_len).saturating_sub(1);
        if out_len == 0 {
            return;
        }

        let mut conv = StreamingPartitionedConvolver::new(ir, channels, partition_size);
        let b = conv.block_size();
        let mut offset = 0usize;

        while offset < out_len {
            let mut dry_block = vec![vec![0.0f32; b]; channels];
            for (ch, block_ch) in dry_block.iter_mut().enumerate().take(channels) {
                let src = input.get(ch).or_else(|| input.first());
                if let Some(src) = src {
                    for (i, sample) in block_ch.iter_mut().enumerate().take(b) {
                        let idx = offset + i;
                        if idx < src.len() {
                            *sample = src[idx];
                        }
                    }
                }
            }

            let wet = conv.process_block(&dry_block);
            let mut mixed = vec![vec![0.0f32; b]; channels];
            for ch in 0..channels {
                for i in 0..b {
                    mixed[ch][i] = dry_block[ch][i] * (1.0 - mix) + wet[ch][i] * mix;
                }
            }

            let valid = (out_len - offset).min(b);
            on_block(&mixed, valid);
            offset += b;
        }
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

    let zero_spec = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut x_history = vec![zero_spec; partitions];
    let mut history_head = 0usize;

    let input_blocks = x.len().div_ceil(b);
    let total_blocks = output_len.div_ceil(b);
    let mut out = vec![0.0f32; output_len];
    let mut overlap = vec![0.0f32; b];
    let inv_fft = 1.0f32 / n_fft as f32;

    for blk in 0..total_blocks {
        let mut x_fft = vec![Complex32::new(0.0, 0.0); n_fft];
        if blk < input_blocks {
            let start = blk * b;
            let end = (start + b).min(x.len());
            for (i, &v) in x[start..end].iter().enumerate() {
                x_fft[i].re = v;
            }
        }
        fft.process(&mut x_fft);
        history_head = (history_head + partitions - 1) % partitions;
        x_history[history_head].copy_from_slice(&x_fft);

        let mut y_fft = vec![Complex32::new(0.0, 0.0); n_fft];
        for (p, h_fft) in h_parts_fft.iter().enumerate() {
            let x_idx = (history_head + p) % partitions;
            let x_hist = &x_history[x_idx];
            for i in 0..n_fft {
                y_fft[i] += x_hist[i] * h_fft[i];
            }
        }
        ifft.process(&mut y_fft);

        let time_offset = blk * b;
        for i in 0..b {
            let out_idx = time_offset + i;
            if out_idx >= output_len {
                break;
            }
            out[out_idx] += y_fft[i].re * inv_fft + overlap[i];
        }

        for i in 0..b {
            overlap[i] = y_fft[i + b].re * inv_fft;
        }
    }

    out
}

struct StreamingPartitionedConvolver {
    channels: usize,
    b: usize,
    n_fft: usize,
    partitions: usize,
    h_parts_fft: Vec<Vec<Vec<Complex32>>>,
    x_history: Vec<Vec<Vec<Complex32>>>,
    overlap: Vec<Vec<f32>>,
    history_head: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl StreamingPartitionedConvolver {
    fn new(ir: &[Vec<f32>], channels: usize, partition_size: usize) -> Self {
        let b = partition_size.max(64);
        let n_fft = (2 * b).next_power_of_two();
        let max_ir_len = ir
            .iter()
            .map(Vec::len)
            .max()
            .or_else(|| ir.first().map(Vec::len))
            .unwrap_or(1)
            .max(1);
        let partitions = max_ir_len.div_ceil(b).max(1);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);

        let mut h_parts_fft =
            vec![vec![vec![Complex32::new(0.0, 0.0); n_fft]; partitions]; channels];
        for (ch, ch_parts) in h_parts_fft.iter_mut().enumerate().take(channels) {
            let h = ir
                .get(ch)
                .or_else(|| ir.first())
                .cloned()
                .unwrap_or_else(|| vec![1.0]);
            for (p, part_fft) in ch_parts.iter_mut().enumerate().take(partitions) {
                let start = p * b;
                let end = (start + b).min(h.len());
                if start >= end {
                    continue;
                }
                let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
                for (i, &v) in h[start..end].iter().enumerate() {
                    buf[i].re = v;
                }
                fft.process(&mut buf);
                *part_fft = buf;
            }
        }

        let x_history = vec![vec![vec![Complex32::new(0.0, 0.0); n_fft]; partitions]; channels];
        let overlap = vec![vec![0.0f32; b]; channels];

        Self {
            channels,
            b,
            n_fft,
            partitions,
            h_parts_fft,
            x_history,
            overlap,
            history_head: 0,
            fft,
            ifft,
        }
    }

    fn block_size(&self) -> usize {
        self.b
    }

    fn process_block(&mut self, dry_block: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut wet_out = vec![vec![0.0f32; self.b]; self.channels];
        let inv_fft = 1.0 / self.n_fft as f32;

        self.history_head = (self.history_head + self.partitions - 1) % self.partitions;

        for ch in 0..self.channels {
            let mut x_fft = vec![Complex32::new(0.0, 0.0); self.n_fft];
            let src = dry_block.get(ch).or_else(|| dry_block.first());
            if let Some(src) = src {
                for (i, &v) in src.iter().take(self.b).enumerate() {
                    x_fft[i].re = v;
                }
            }
            self.fft.process(&mut x_fft);
            self.x_history[ch][self.history_head].copy_from_slice(&x_fft);

            let mut y_fft = vec![Complex32::new(0.0, 0.0); self.n_fft];
            for p in 0..self.partitions {
                let x_idx = (self.history_head + p) % self.partitions;
                let x_hist = &self.x_history[ch][x_idx];
                let h_fft = &self.h_parts_fft[ch][p];
                for i in 0..self.n_fft {
                    y_fft[i] += x_hist[i] * h_fft[i];
                }
            }
            self.ifft.process(&mut y_fft);

            for i in 0..self.b {
                wet_out[ch][i] = y_fft[i].re * inv_fft + self.overlap[ch][i];
                self.overlap[ch][i] = y_fft[i + self.b].re * inv_fft;
            }
        }

        wet_out
    }
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
