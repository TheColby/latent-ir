# Learned Encoders

`latent-ir` supports lightweight learned conditioning models that produce descriptor deltas.

These models are intentionally inspectable and deterministic in v0.x.

## Supported Model Families

- `LearnedTextEncoderModel`
- `LearnedAudioEncoderModel`

Optional ONNX runtime adapters are available behind the `onnx` cargo feature.

## Text Encoder

Typical model fields:

- `model_version`
- `embedding_dim`
- `token_embeddings`
- `unknown_embedding`
- `projection`
- `bias`
- `output_scale`

Inference flow:

1. tokenize prompt
2. aggregate token embeddings
3. project to descriptor delta
4. apply delta to `DescriptorSet`

## Audio Encoder

Typical model fields:

- `model_version`
- `feature_names`
- `input_mean`, `input_std`
- `hidden_weights`, `hidden_bias`
- `projection`, `bias`, `output_scale`

Current engineered feature frontend includes duration/peak/RMS/ZCR/predelay and coarse spectral-energy summaries.

Inference flow:

1. extract features from reference audio
2. normalize feature vector
3. run hidden transform
4. project to descriptor delta
5. apply delta to `DescriptorSet`

## ONNX Inference

Enable ONNX path:

```bash
cargo run --features onnx -- generate ...
```

Current scaffold assumptions:

- text ONNX input: `[1, input_dim]` hashed prompt feature vector
- audio ONNX input: `[1, 10]` engineered feature vector
- output: descriptor delta vector (canonical field order)

## Training Commands

- `train-encoder text`
- `train-encoder audio`

Text dataset format:

```json
[
  {
    "prompt": "dark stone cathedral",
    "descriptor": { "...": "..." }
  }
]
```

Audio dataset format:

```json
[
  {
    "audio_path": "relative/or/absolute.wav",
    "descriptor": { "...": "..." }
  }
]
```

Relative `audio_path` values are resolved from the dataset file directory.

Industrial starter set included:

- `examples/datasets/text_pairs_industrial.json`

## Evaluation Commands

- `eval text`
- `eval audio`
- `eval check`

Baseline report schema:

- `latent-ir.eval.baseline.v1`

Core sections:

- descriptor MAE/RMSE
- analysis MAE/RMSE
- per-field/per-metric detail maps

## Scope Notes

- These are not large pretrained transformer encoders.
- DSP fallback and explicit overrides remain first-class.
- The learned path is additive and auditable, not opaque replacement logic.
