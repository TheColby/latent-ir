use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DescriptorSet {
    pub time: TimeDescriptors,
    pub spectral: SpectralDescriptors,
    pub structural: StructuralDescriptors,
    pub spatial: SpatialDescriptors,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeDescriptors {
    pub duration: f32,
    pub predelay_ms: f32,
    pub t60: f32,
    pub edt: f32,
    pub attack_gap_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpectralDescriptors {
    pub brightness: f32,
    pub hf_damping: f32,
    pub lf_bloom: f32,
    pub spectral_tilt: f32,
    pub band_decay_low: f32,
    pub band_decay_mid: f32,
    pub band_decay_high: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuralDescriptors {
    pub early_density: f32,
    pub late_density: f32,
    pub diffusion: f32,
    pub modal_density: f32,
    pub tail_noise: f32,
    pub grain: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpatialDescriptors {
    pub channel_format: ChannelFormat,
    pub width: f32,
    pub decorrelation: f32,
    pub asymmetry: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChannelFormat {
    Mono,
    Stereo,
}

impl ChannelFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

impl Default for DescriptorSet {
    fn default() -> Self {
        Self {
            time: TimeDescriptors {
                duration: 3.5,
                predelay_ms: 18.0,
                t60: 2.4,
                edt: 1.8,
                attack_gap_ms: 4.0,
            },
            spectral: SpectralDescriptors {
                brightness: 0.55,
                hf_damping: 0.45,
                lf_bloom: 0.45,
                spectral_tilt: -0.2,
                band_decay_low: 1.2,
                band_decay_mid: 1.0,
                band_decay_high: 0.72,
            },
            structural: StructuralDescriptors {
                early_density: 0.38,
                late_density: 0.72,
                diffusion: 0.62,
                modal_density: 0.55,
                tail_noise: 0.25,
                grain: 0.2,
            },
            spatial: SpatialDescriptors {
                channel_format: ChannelFormat::Stereo,
                width: 0.72,
                decorrelation: 0.4,
                asymmetry: 0.0,
            },
        }
    }
}

impl DescriptorSet {
    pub fn apply_overrides(
        &mut self,
        duration: Option<f32>,
        t60: Option<f32>,
        predelay_ms: Option<f32>,
        edt: Option<f32>,
    ) {
        if let Some(v) = duration {
            self.time.duration = v;
        }
        if let Some(v) = t60 {
            self.time.t60 = v;
        }
        if let Some(v) = predelay_ms {
            self.time.predelay_ms = v;
        }
        if let Some(v) = edt {
            self.time.edt = v;
        }
    }

    pub fn apply_spectral_overrides(
        &mut self,
        brightness: Option<f32>,
        hf_damping: Option<f32>,
        lf_bloom: Option<f32>,
        spectral_tilt: Option<f32>,
    ) {
        if let Some(v) = brightness {
            self.spectral.brightness = v;
        }
        if let Some(v) = hf_damping {
            self.spectral.hf_damping = v;
        }
        if let Some(v) = lf_bloom {
            self.spectral.lf_bloom = v;
        }
        if let Some(v) = spectral_tilt {
            self.spectral.spectral_tilt = v;
        }
    }

    pub fn apply_structure_overrides(
        &mut self,
        early_density: Option<f32>,
        late_density: Option<f32>,
        diffusion: Option<f32>,
    ) {
        if let Some(v) = early_density {
            self.structural.early_density = v;
        }
        if let Some(v) = late_density {
            self.structural.late_density = v;
        }
        if let Some(v) = diffusion {
            self.structural.diffusion = v;
        }
    }

    pub fn apply_spatial_overrides(
        &mut self,
        channel_format: Option<ChannelFormat>,
        width: Option<f32>,
        decorrelation: Option<f32>,
        asymmetry: Option<f32>,
    ) {
        if let Some(v) = channel_format {
            self.spatial.channel_format = v;
        }
        if let Some(v) = width {
            self.spatial.width = v;
        }
        if let Some(v) = decorrelation {
            self.spatial.decorrelation = v;
        }
        if let Some(v) = asymmetry {
            self.spatial.asymmetry = v;
        }
    }

    pub fn clamp(&mut self) {
        self.time.duration = self.time.duration.clamp(0.1, 30.0);
        self.time.predelay_ms = self.time.predelay_ms.clamp(0.0, 500.0);
        self.time.t60 = self.time.t60.clamp(0.1, 60.0);
        self.time.edt = self.time.edt.clamp(0.05, 30.0);
        self.time.attack_gap_ms = self.time.attack_gap_ms.clamp(0.0, 100.0);

        self.spectral.brightness = self.spectral.brightness.clamp(0.0, 1.0);
        self.spectral.hf_damping = self.spectral.hf_damping.clamp(0.0, 1.0);
        self.spectral.lf_bloom = self.spectral.lf_bloom.clamp(0.0, 1.0);
        self.spectral.spectral_tilt = self.spectral.spectral_tilt.clamp(-2.0, 2.0);
        self.spectral.band_decay_low = self.spectral.band_decay_low.clamp(0.2, 2.5);
        self.spectral.band_decay_mid = self.spectral.band_decay_mid.clamp(0.2, 2.5);
        self.spectral.band_decay_high = self.spectral.band_decay_high.clamp(0.2, 2.5);

        self.structural.early_density = self.structural.early_density.clamp(0.0, 1.0);
        self.structural.late_density = self.structural.late_density.clamp(0.0, 1.0);
        self.structural.diffusion = self.structural.diffusion.clamp(0.0, 1.0);
        self.structural.modal_density = self.structural.modal_density.clamp(0.0, 1.0);
        self.structural.tail_noise = self.structural.tail_noise.clamp(0.0, 1.0);
        self.structural.grain = self.structural.grain.clamp(0.0, 1.0);

        self.spatial.width = self.spatial.width.clamp(0.0, 1.0);
        self.spatial.decorrelation = self.spatial.decorrelation.clamp(0.0, 1.0);
        self.spatial.asymmetry = self.spatial.asymmetry.clamp(-1.0, 1.0);
    }
}
