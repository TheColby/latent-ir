use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::core::conditioning::{
    AudioEncoder, LearnedAudioEncoder, LearnedTextEncoder, TextEncoder,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    TextJsonV1,
    AudioJsonV1,
    TextOnnxV1,
    AudioOnnxV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: String,
    pub name: String,
    pub format: ModelFormat,
    pub model_path: PathBuf,
    pub input_dim: usize,
    pub output_dim: usize,
    #[serde(default)]
    pub deterministic: bool,
    #[serde(default)]
    pub required_features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCapabilities {
    pub onnx_enabled: bool,
    pub descriptor_output_dim: usize,
}

impl RuntimeCapabilities {
    pub fn current() -> Self {
        Self {
            onnx_enabled: cfg!(feature = "onnx"),
            descriptor_output_dim: 20,
        }
    }
}

pub fn load_manifest(path: impl AsRef<Path>) -> Result<ModelManifest> {
    let text = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read manifest {}", path.as_ref().display()))?;
    serde_json::from_str(&text).with_context(|| "failed to parse model manifest JSON")
}

pub fn validate_manifest(manifest: &ModelManifest, runtime: &RuntimeCapabilities) -> Result<()> {
    if manifest.schema_version != "latent-ir.model-manifest.v1" {
        return Err(anyhow!(
            "unsupported manifest schema_version '{}'",
            manifest.schema_version
        ));
    }
    if manifest.output_dim != runtime.descriptor_output_dim {
        return Err(anyhow!(
            "manifest output_dim={} must match runtime descriptor_output_dim={}",
            manifest.output_dim,
            runtime.descriptor_output_dim
        ));
    }

    let requires_onnx = matches!(
        manifest.format,
        ModelFormat::TextOnnxV1 | ModelFormat::AudioOnnxV1
    ) || manifest.required_features.iter().any(|f| f == "onnx");
    if requires_onnx && !runtime.onnx_enabled {
        return Err(anyhow!(
            "model requires onnx feature; rebuild with `--features onnx`"
        ));
    }

    match manifest.format {
        ModelFormat::TextJsonV1 => {
            let model = LearnedTextEncoder::from_json_file(&manifest.model_path)?;
            let _ = model.infer_delta_from_prompt("sanity prompt")?;
            if manifest.input_dim == 0 {
                return Err(anyhow!("text json model input_dim must be > 0"));
            }
        }
        ModelFormat::AudioJsonV1 => {
            let model = LearnedAudioEncoder::from_json_file(&manifest.model_path)?;
            let dummy = vec![vec![0.0f32; 512]];
            let _ = model.infer_delta_from_audio(&dummy, 48_000)?;
            if manifest.input_dim != 10 {
                return Err(anyhow!(
                    "audio json model input_dim should be 10 (engineered feature vector), got {}",
                    manifest.input_dim
                ));
            }
        }
        ModelFormat::TextOnnxV1 | ModelFormat::AudioOnnxV1 => {
            if !manifest.model_path.exists() {
                return Err(anyhow!(
                    "onnx model file does not exist: {}",
                    manifest.model_path.display()
                ));
            }
        }
    }

    Ok(())
}
