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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_position_m: Option<CartesianPosition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listener_position_m: Option<CartesianPosition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_layout: Option<CustomChannelLayout>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChannelFormat {
    Mono,
    Stereo,
    FoaAmbix,
    Surround5_1,
    Surround7_1,
    Atmos7_1_4,
    Atmos7_2_4,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChannelSpec {
    pub label: String,
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    #[serde(default)]
    pub is_lfe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_m: Option<CartesianPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomChannelLayout {
    pub layout_name: String,
    #[serde(default)]
    pub spatial_encoding: SpatialEncoding,
    pub channels: Vec<ChannelSpec>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CartesianPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpatialEncoding {
    #[default]
    Discrete,
    AmbisonicFoaAmbix,
}

impl SpatialEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::AmbisonicFoaAmbix => "ambisonic_foa_ambix",
        }
    }

    pub fn from_str_name(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "discrete" => Some(Self::Discrete),
            "ambisonic_foa_ambix" | "foa" | "ambix" => Some(Self::AmbisonicFoaAmbix),
            _ => None,
        }
    }
}

impl ChannelFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::FoaAmbix => 4,
            Self::Surround5_1 => 6,
            Self::Surround7_1 => 8,
            Self::Atmos7_1_4 => 12,
            Self::Atmos7_2_4 => 13,
            Self::Custom => 0,
        }
    }

    pub fn layout_name(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Stereo => "stereo",
            Self::FoaAmbix => "foa_ambix",
            Self::Surround5_1 => "surround_5_1",
            Self::Surround7_1 => "surround_7_1",
            Self::Atmos7_1_4 => "atmos_7_1_4",
            Self::Atmos7_2_4 => "atmos_7_2_4",
            Self::Custom => "custom",
        }
    }

    pub fn spatial_encoding(self) -> SpatialEncoding {
        match self {
            Self::FoaAmbix => SpatialEncoding::AmbisonicFoaAmbix,
            _ => SpatialEncoding::Discrete,
        }
    }

    pub fn builtin_channel_specs(self) -> Option<Vec<ChannelSpec>> {
        let specs = match self {
            Self::Mono => vec![ch("M", 0, 0, false)],
            Self::Stereo => vec![ch("L", 30, 0, false), ch("R", -30, 0, false)],
            Self::FoaAmbix => vec![
                ch("W", 0, 0, false),
                ch("X", 0, 0, false),
                ch("Y", 0, 0, false),
                ch("Z", 0, 0, false),
            ],
            Self::Surround5_1 => vec![
                ch("L", 30, 0, false),
                ch("R", -30, 0, false),
                ch("C", 0, 0, false),
                ch("LFE", 0, 0, true),
                ch("Ls", 110, 0, false),
                ch("Rs", -110, 0, false),
            ],
            Self::Surround7_1 => vec![
                ch("L", 30, 0, false),
                ch("R", -30, 0, false),
                ch("C", 0, 0, false),
                ch("LFE", 0, 0, true),
                ch("Ls", 90, 0, false),
                ch("Rs", -90, 0, false),
                ch("Lrs", 150, 0, false),
                ch("Rrs", -150, 0, false),
            ],
            Self::Atmos7_1_4 => vec![
                ch("L", 30, 0, false),
                ch("R", -30, 0, false),
                ch("C", 0, 0, false),
                ch("LFE", 0, 0, true),
                ch("Ls", 90, 0, false),
                ch("Rs", -90, 0, false),
                ch("Lrs", 150, 0, false),
                ch("Rrs", -150, 0, false),
                ch("Ltf", 35, 45, false),
                ch("Rtf", -35, 45, false),
                ch("Ltr", 145, 45, false),
                ch("Rtr", -145, 45, false),
            ],
            Self::Atmos7_2_4 => vec![
                ch("L", 30, 0, false),
                ch("R", -30, 0, false),
                ch("C", 0, 0, false),
                ch("LFE1", 0, 0, true),
                ch("LFE2", 0, 0, true),
                ch("Ls", 90, 0, false),
                ch("Rs", -90, 0, false),
                ch("Lrs", 150, 0, false),
                ch("Rrs", -150, 0, false),
                ch("Ltf", 35, 45, false),
                ch("Rtf", -35, 45, false),
                ch("Ltr", 145, 45, false),
                ch("Rtr", -145, 45, false),
            ],
            Self::Custom => return None,
        };
        Some(specs)
    }
}

impl SpatialDescriptors {
    pub fn set_custom_layout(&mut self, layout: CustomChannelLayout) {
        self.channel_format = ChannelFormat::Custom;
        self.custom_layout = Some(layout);
    }

    pub fn clear_custom_layout(&mut self) {
        self.custom_layout = None;
    }

    pub fn resolved_layout_name(&self) -> String {
        match self.channel_format {
            ChannelFormat::Custom => self
                .custom_layout
                .as_ref()
                .map(|l| l.layout_name.clone())
                .unwrap_or_else(|| "custom".to_string()),
            f => f.layout_name().to_string(),
        }
    }

    pub fn resolved_spatial_encoding(&self) -> SpatialEncoding {
        match self.channel_format {
            ChannelFormat::Custom => self
                .custom_layout
                .as_ref()
                .map(|l| l.spatial_encoding)
                .unwrap_or(SpatialEncoding::Discrete),
            f => f.spatial_encoding(),
        }
    }

    pub fn resolved_channel_specs(&self) -> Vec<ChannelSpec> {
        match self.channel_format {
            ChannelFormat::Custom => self
                .custom_layout
                .as_ref()
                .map(|l| l.channels.clone())
                .unwrap_or_default(),
            f => f.builtin_channel_specs().unwrap_or_default(),
        }
    }

    pub fn resolved_channel_labels(&self) -> Vec<String> {
        self.resolved_channel_specs()
            .into_iter()
            .map(|c| c.label)
            .collect()
    }

    pub fn resolved_channels(&self) -> usize {
        self.resolved_channel_specs().len()
    }
}

fn ch(label: &str, azimuth_deg: i16, elevation_deg: i16, is_lfe: bool) -> ChannelSpec {
    ChannelSpec {
        label: label.to_string(),
        azimuth_deg: azimuth_deg as f32,
        elevation_deg: elevation_deg as f32,
        is_lfe,
        position_m: None,
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
                source_position_m: None,
                listener_position_m: None,
                custom_layout: None,
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
            if v != ChannelFormat::Custom {
                self.spatial.clear_custom_layout();
            }
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
