# Model Manifests

Model manifests define runtime compatibility contracts for learned models.

Schema:
- `latent-ir.model-manifest.v1`

## Why Manifests Exist

- explicit model/runtime compatibility checks
- safer model swapping in CI and scripts
- stronger reproducibility for experiments and releases

## Core Fields

- `name`: identifier
- `format`:
  - `text_json_v1`
  - `audio_json_v1`
  - `text_onnx_v1`
  - `audio_onnx_v1`
- `model_path`: model file location
- `input_dim`: declared input width
- `output_dim`: descriptor delta width (currently `20`)
- `deterministic`: deterministic inference declaration
- `required_features`: runtime features (for example `onnx`)

## Validation Command

```bash
cargo run -- model validate --manifest manifests/text_encoder_manifest.json
```

Validation checks include:
- schema/version compatibility
- output-dimension compatibility with descriptor contract
- feature requirements
- format-specific constraints

## Deployment Recommendation

For reproducible pipelines:
1. pin model files in version control or artifact storage
2. validate manifests in CI
3. include manifest + model path in experiment metadata
