use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use serde_json::json;
use tempfile::tempdir;

#[test]
fn model_validate_passes_for_text_json_manifest() {
    let dir = tempdir().expect("tempdir");
    let manifest = dir.path().join("text_manifest.json");

    let m = json!({
      "schema_version": "latent-ir.model-manifest.v1",
      "name": "text-json-sample",
      "format": "text_json_v1",
      "model_path": "examples/models/text_encoder_v1.json",
      "input_dim": 6,
      "output_dim": 20,
      "deterministic": true,
      "required_features": []
    });
    std::fs::write(&manifest, serde_json::to_string_pretty(&m).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "model",
        "validate",
        "--manifest",
        manifest.to_str().unwrap(),
    ])
    .unwrap();

    dispatch(cli).expect("manifest should validate");
}

#[test]
fn model_validate_fails_for_missing_onnx_feature() {
    let dir = tempdir().expect("tempdir");
    let manifest = dir.path().join("onnx_manifest.json");

    let m = json!({
      "schema_version": "latent-ir.model-manifest.v1",
      "name": "text-onnx-sample",
      "format": "text_onnx_v1",
      "model_path": "missing.onnx",
      "input_dim": 256,
      "output_dim": 20,
      "deterministic": true,
      "required_features": ["onnx"]
    });
    std::fs::write(&manifest, serde_json::to_string_pretty(&m).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "model",
        "validate",
        "--manifest",
        manifest.to_str().unwrap(),
    ])
    .unwrap();

    #[cfg(not(feature = "onnx"))]
    assert!(dispatch(cli).is_err());

    #[cfg(feature = "onnx")]
    assert!(dispatch(cli).is_err());
}
