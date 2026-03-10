# Model Manifests

`latent-ir` uses model manifests to declare runtime compatibility and enforce stable contracts.

## Schema

`schema_version`: `latent-ir.model-manifest.v1`

Fields:

- `name`: model identifier
- `format`: one of
  - `text_json_v1`
  - `audio_json_v1`
  - `text_onnx_v1`
  - `audio_onnx_v1`
- `model_path`: path to model file
- `input_dim`: declared input dimension
- `output_dim`: must be `20` (descriptor delta size)
- `deterministic`: boolean
- `required_features`: e.g. `["onnx"]`

## Validate command

```bash
cargo run -- model validate --manifest manifests/text_encoder_manifest.json
```

Validation checks:

- schema version compatibility
- output dimension compatibility with runtime descriptor contract
- required feature availability (e.g. `onnx`)
- format-specific structural checks
