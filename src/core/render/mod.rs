#[derive(Debug, Default, Clone)]
pub struct Renderer;

impl Renderer {
    pub fn render_convolution(
        &self,
        input: &[Vec<f32>],
        ir: &[Vec<f32>],
        mix: f32,
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
            wet.push(convolve(&in_ch, &ir_ch));
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

fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
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
