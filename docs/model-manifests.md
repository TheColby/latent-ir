# Model Manifests

Model manifests declare runtime compatibility contracts for learned models.

Schema version:

- `latent-ir.model-manifest.v1`

## Manifest Fields

- `name`: model identifier
- `format`:
  - `text_json_v1`
  - `audio_json_v1`
  - `text_onnx_v1`
  - `audio_onnx_v1`
- `model_path`: model file path
- `input_dim`: declared input width
- `output_dim`: descriptor delta width (currently `20`)
- `deterministic`: whether inference is deterministic
- `required_features`: runtime feature requirements (for example `"onnx"`)

## Validation Command

```bash
cargo run -- model validate --manifest manifests/text_encoder_manifest.json
```

Validation checks include:

- schema/version compatibility
- output-dimension compatibility with runtime descriptor contract
- required feature availability
- format-specific constraints

## Why Use Manifests

- explicit runtime contract for models
- safer model swapping in scripts/CI
- better long-term reproducibility for experiments and releases
