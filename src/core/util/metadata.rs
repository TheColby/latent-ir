use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::analysis::AnalysisReport;
use crate::core::conditioning::{ConditioningUncertainty, DescriptorDelta};
use crate::core::descriptors::DescriptorSet;
use crate::core::perceptual::MacroControls;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub schema_version: String,
    pub project: String,
    pub version: String,
    pub command: String,
    pub replay_command: String,
    pub seed: u64,
    pub prompt: Option<String>,
    pub preset: Option<String>,
    pub conditioning: ConditioningTrace,
    pub sample_rate: u32,
    pub spatial_encoding: String,
    pub channel_format: String,
    pub channel_labels: Vec<String>,
    pub channel_map_path: Option<String>,
    pub ir_sha256: String,
    pub descriptor_sha256: String,
    pub channel_map_sha256: String,
    pub descriptor: DescriptorSet,
    pub quality_gate_profile: Option<String>,
    pub quality_gate_passed: Option<bool>,
    pub quality_gate_failed_checks: Option<Vec<String>>,
    pub warnings: Vec<String>,
    pub generated_at_utc: DateTime<Utc>,
    pub analysis: AnalysisReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConditioningTrace {
    pub text_encoder_model: Option<String>,
    pub text_encoder_onnx: Option<String>,
    pub audio_encoder_model: Option<String>,
    pub audio_encoder_onnx: Option<String>,
    pub reference_audio: Option<String>,
    pub text_delta: Option<DescriptorDelta>,
    pub audio_delta: Option<DescriptorDelta>,
    pub combined_delta: Option<DescriptorDelta>,
    pub uncertainty: Option<ConditioningUncertainty>,
    pub macro_controls: Option<MacroControls>,
    pub macro_trajectory: Option<String>,
}

pub fn companion_json_path(audio_path: &Path) -> PathBuf {
    let mut p = audio_path.to_path_buf();
    p.set_extension("json");
    p
}
