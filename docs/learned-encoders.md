# Learned Encoders

`latent-ir` supports learned descriptor conditioning through two model types loaded from JSON:

- `LearnedTextEncoderModel`
- `LearnedAudioEncoderModel`

These are intentionally lightweight and fully inspectable for v0.1.

The project also supports ONNX inference backends behind the cargo feature flag `onnx`.

## Text Encoder

Model fields:

- `model_version`
- `embedding_dim`
- `token_embeddings`: token -> embedding vector
- `unknown_embedding`
- `projection`: linear projection from embedding -> descriptor deltas
- `bias`
- `output_scale`

Inference flow:

1. tokenize prompt
2. average token embeddings
3. project embedding to `DescriptorDelta`
4. apply delta to `DescriptorSet`

## Audio Encoder

Model fields:

- `model_version`
- `feature_names`
- `input_mean`, `input_std`
- `hidden_weights`, `hidden_bias` (single hidden tanh layer)
- `projection`, `bias`, `output_scale`

Audio frontend features (v0.1):

- duration
- peak
- RMS
- zero crossing rate
- predelay estimate (normalized)
- early energy ratio
- low/mid/high energy ratios
- normalized centroid proxy

Inference flow:

1. extract feature vector from reference audio
2. normalize features
3. run hidden layer + tanh
4. project hidden state to `DescriptorDelta`
5. apply delta to `DescriptorSet`

## Scope Notes

- These are learned weight paths, not large foundation models.
- Models are deterministic and file-driven.
- The CLI still supports rule-based semantics and explicit overrides; those remain first-class and composable.

## ONNX Inference

Enable with:

```bash
cargo run --features onnx -- generate ...
```

ONNX assumptions in current scaffold:

- text ONNX model input: single `[1, input_dim]` hashed token feature vector
- audio ONNX model input: single `[1, 10]` engineered audio feature vector
- output: at least 20 floats mapping to descriptor deltas in canonical field order

## Training Utility

`latent-ir` includes:

- `train-encoder text`
- `train-encoder audio`

### Text dataset format

JSON array:

```json
[
  {
    "prompt": "dark stone cathedral",
    "descriptor": { "... DescriptorSet fields ..." : "..." }
  }
]
```

### Audio dataset format

JSON array:

```json
[
  {
    "audio_path": "relative/or/absolute/path.wav",
    "descriptor": { "... DescriptorSet fields ..." : "..." }
  }
]
```

Audio paths are resolved relative to the dataset file when not absolute.

### Industrial text-conditioning starter dataset

This repo includes:

- `examples/datasets/text_pairs_industrial.json`

It focuses on industrial structures/materials (silos, bunkers, cisterns, hangars, corrugated steel, poured concrete).

Train from it with:

```bash
cargo run -- train-encoder text \
  --dataset examples/datasets/text_pairs_industrial.json \
  --output models/text_encoder_industrial_v1.json \
  --max-vocab 512 \
  --epochs 1200
```

## Evaluation Utility

Use `eval text` or `eval audio` to generate baseline reports for held-out sets.

Baseline report schema:

- `schema_version`: `latent-ir.eval.baseline.v1`
- `sample_count`
- `descriptor_metrics`:
  - `mae`
  - `rmse`
  - `per_field_mae`
- `analysis_metrics`:
  - `mae`
  - `rmse`
  - `per_metric_mae`

This report is intended to be checked in or archived so future model/generator changes can be compared against a stable reference.
