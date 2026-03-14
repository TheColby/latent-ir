use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::descriptors::DescriptorSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: String,
    pub project: String,
    pub version: String,
    pub generated_at_utc: DateTime<Utc>,
    pub config: DatasetConfigSnapshot,
    pub summary: DatasetSummary,
    pub records: Vec<DatasetRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfigSnapshot {
    pub count_requested: usize,
    pub count_succeeded: usize,
    pub count_failed: usize,
    pub seed: u64,
    pub sample_rate: u32,
    pub channel_format: String,
    pub duration_range_s: [f32; 2],
    pub t60_range_s: [f32; 2],
    pub predelay_max_ms: f32,
    pub jitter: f32,
    pub preset_mix: f32,
    pub prompt_bank_size: usize,
    pub quality_gate: bool,
    pub quality_profile: Option<String>,
    pub tail_fade_ms: Option<f32>,
    pub export_training_json: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub mean_duration_s: f32,
    pub mean_t60_s_est: f32,
    pub mean_predelay_ms_est: f32,
    pub mean_decay_db_span: f32,
    pub quality_gate_failures: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub id: String,
    pub seed: u64,
    pub prompt: String,
    pub preset: Option<String>,
    pub ir_wav: String,
    pub metadata_json: String,
    pub analysis_json: String,
    pub channel_map_json: Option<String>,
    pub descriptor: DescriptorSet,
    pub duration_s: f32,
    pub t60_s_est: Option<f32>,
    pub predelay_ms_est: f32,
    pub decay_db_span: f32,
    pub quality_gate_passed: Option<bool>,
}
