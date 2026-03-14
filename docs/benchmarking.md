# Benchmarking and Regression

`latent-ir` includes evaluation and benchmark tooling intended for reproducible regression control.

## Components

- `eval text` / `eval audio`: baseline report generation for learned conditioning models
- `eval check`: regression gate against committed baseline reports
- `benchmark run`: end-to-end objective/speed/stability/perceptual-proxy report
- `benchmark check`: regression gate for benchmark reports
- spatial corpus tests: envelope checks for multichannel spatial behavior
- streaming render QA: numerical parity and block-boundary integrity checks

## Benchmark Dataset Schema

`benchmark run` expects JSON with schema `latent-ir.benchmark.dataset.v1`:

```json
{
  "schema_version": "latent-ir.benchmark.dataset.v1",
  "samples": [
    {
      "id": "sample_001",
      "prompt": "dark cathedral",
      "reference_audio": "refs/hall_ir.wav",
      "target_ir": "targets/hall_target.wav",
      "target_descriptor": { "...": "..." }
    }
  ]
}
```

Only `id` is strictly required; quality improves when target descriptor and/or target IR are provided.

## Run Benchmark

```bash
cargo run -- benchmark run \
  --dataset datasets/benchmark_suite.json \
  --text-model models/text_encoder_v1.json \
  --audio-model models/audio_encoder_v1.json \
  --repeats 3 \
  --output reports/benchmark.json
```

Report schema: `latent-ir.benchmark.v1`.

## Regression Checks

```bash
cargo run -- eval check \
  --report reports/eval_text_new.json \
  --baseline ci/baselines/eval_text_baseline.json \
  --max-regression 0.10

cargo run -- benchmark check \
  --report reports/benchmark_new.json \
  --baseline ci/baselines/benchmark_baseline.json \
  --max-regression 0.05
```

Checks fail when selected key metrics regress past thresholds.

## Spatial Regression Corpus

Dataset:

- `ci/datasets/spatial_corpus_ci.json`

Test harness:

- `tests/spatial_corpus_tests.rs`

Coverage includes:

- 7.2.4 bed
- custom 16-channel ring
- custom A.B.C-style elevated layout
- cartesian-only custom layout ingestion

Envelope metrics include:

- inter-channel mean absolute correlation
- directional/LFE energy ratios

## Render QA

Render QA tests:

- `tests/render_spatial_qa_tests.rs`

Checks include:

- direct vs streaming numerical parity
- output length correctness
- block-boundary artifact detection

## CI Entry Point

Run all local gates:

```bash
./scripts/ci_regression_gates.sh
```

This script mirrors repository CI behavior (`.github/workflows/regression-gates.yml`).
