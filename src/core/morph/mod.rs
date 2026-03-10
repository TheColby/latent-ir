use crate::core::descriptors::DescriptorSet;

#[derive(Debug, Default, Clone)]
pub struct IrMorpher;

impl IrMorpher {
    pub fn morph(&self, a: &[Vec<f32>], b: &[Vec<f32>], alpha: f32) -> Vec<Vec<f32>> {
        let alpha = alpha.clamp(0.0, 1.0);
        let channels = a.len().max(b.len());
        let len = a
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(b.iter().map(Vec::len).max().unwrap_or(0));
        let mut out = vec![vec![0.0; len]; channels];

        for ch in 0..channels {
            let a_ch = a.get(ch).or_else(|| a.first());
            let b_ch = b.get(ch).or_else(|| b.first());
            for i in 0..len {
                let av = a_ch.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                let bv = b_ch.and_then(|v| v.get(i)).copied().unwrap_or(0.0);
                out[ch][i] = av * (1.0 - alpha) + bv * alpha;
            }
        }

        normalize(&mut out);
        out
    }

    pub fn morph_descriptors(
        &self,
        a: &DescriptorSet,
        b: &DescriptorSet,
        alpha: f32,
    ) -> DescriptorSet {
        let alpha = alpha.clamp(0.0, 1.0);
        let lerp = |x: f32, y: f32| x * (1.0 - alpha) + y * alpha;

        let mut out = a.clone();
        out.time.duration = lerp(a.time.duration, b.time.duration);
        out.time.predelay_ms = lerp(a.time.predelay_ms, b.time.predelay_ms);
        out.time.t60 = lerp(a.time.t60, b.time.t60);
        out.time.edt = lerp(a.time.edt, b.time.edt);
        out.time.attack_gap_ms = lerp(a.time.attack_gap_ms, b.time.attack_gap_ms);

        out.spectral.brightness = lerp(a.spectral.brightness, b.spectral.brightness);
        out.spectral.hf_damping = lerp(a.spectral.hf_damping, b.spectral.hf_damping);
        out.spectral.lf_bloom = lerp(a.spectral.lf_bloom, b.spectral.lf_bloom);
        out.spectral.spectral_tilt = lerp(a.spectral.spectral_tilt, b.spectral.spectral_tilt);
        out.spectral.band_decay_low = lerp(a.spectral.band_decay_low, b.spectral.band_decay_low);
        out.spectral.band_decay_mid = lerp(a.spectral.band_decay_mid, b.spectral.band_decay_mid);
        out.spectral.band_decay_high = lerp(a.spectral.band_decay_high, b.spectral.band_decay_high);

        out.structural.early_density = lerp(a.structural.early_density, b.structural.early_density);
        out.structural.late_density = lerp(a.structural.late_density, b.structural.late_density);
        out.structural.diffusion = lerp(a.structural.diffusion, b.structural.diffusion);
        out.structural.modal_density = lerp(a.structural.modal_density, b.structural.modal_density);
        out.structural.tail_noise = lerp(a.structural.tail_noise, b.structural.tail_noise);
        out.structural.grain = lerp(a.structural.grain, b.structural.grain);

        out.spatial.width = lerp(a.spatial.width, b.spatial.width);
        out.spatial.decorrelation = lerp(a.spatial.decorrelation, b.spatial.decorrelation);
        out.spatial.asymmetry = lerp(a.spatial.asymmetry, b.spatial.asymmetry);
        out
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
    let gain = 0.98 / peak;
    for ch in channels {
        for s in ch {
            *s *= gain;
        }
    }
}
