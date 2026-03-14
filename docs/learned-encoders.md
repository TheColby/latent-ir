# Learned Encoders

`latent-ir` supports lightweight learned conditioning models that emit descriptor deltas.

These models are intentionally inspectable, deterministic, and local-first in v0.x.

## Important Scope

- This is not a giant pretrained transformer stack in the generation path.
- Learned modules currently modify descriptors before DSP synthesis.
- No external AI service key is required for JSON-model workflows.
- Conditioning traces are emitted to generation metadata (including combined total delta).

## Supported Families

- `LearnedTextEncoderModel`
- `LearnedAudioEncoderModel`

Optional ONNX adapters are available via the `onnx` feature.

## Text Encoder Shape

Typical fields:
- `model_version`
- `embedding_dim`
- `token_embeddings`
- `unknown_embedding`
- `projection`
- `bias`
- `output_scale`

Inference flow:
1. tokenize prompt
2. embed/aggregate
3. project to descriptor delta
4. apply to `DescriptorSet`

## Audio Encoder Shape

Typical fields:
- `model_version`
- `feature_names`
- `input_mean`, `input_std`
- `hidden_weights`, `hidden_bias`
- `projection`, `bias`, `output_scale`

Current feature frontend is engineered DSP statistics (duration/peak/RMS/ZCR/predelay/coarse spectral summaries).

Inference flow:
1. extract features from reference audio
2. normalize feature vector
3. hidden transform
4. projection to descriptor delta
5. apply to `DescriptorSet`

## ONNX Runtime Notes

Enable ONNX paths:

```bash
cargo run --features onnx -- generate ...
```

Current assumptions:
- text ONNX input: `[1, input_dim]` hashed prompt frontend
- audio ONNX input: `[1, 10]` engineered feature frontend
- output: 20-value descriptor delta vector

## Training Commands

- `train-encoder text`
- `train-encoder audio`

Text dataset example:

```json
[
  {
    "prompt": "dark stone cathedral",
    "descriptor": { "...": "..." }
  }
]
```

Audio dataset example:

```json
[
  {
    "audio_path": "relative/or/absolute.wav",
    "descriptor": { "...": "..." }
  }
]
```

Relative `audio_path` values are resolved from dataset directory.

## Evaluation Commands

- `eval text`
- `eval audio`
- `eval check`

Baseline schema:
- `latent-ir.eval.baseline.v1`

## Manifest Integration

Pair learned models with model manifests (`docs/model-manifests.md`) for runtime compatibility checks and reproducible deployment.
