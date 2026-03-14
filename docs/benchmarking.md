# Benchmarking and Regression

`latent-ir` ships with reproducible evaluation and regression tooling intended to catch real behavior drift before release.

## Tooling Surface

- `eval text`
- `eval audio`
- `eval check`
- `benchmark run`
- `benchmark check`
- spatial corpus tests
- streaming render QA tests

## What Each Tool Does

- `eval text` / `eval audio`
  - evaluates learned conditioning quality against labeled datasets
  - emits baseline-friendly reports
- `eval check`
  - compares new eval report to committed baseline
  - fails on configured regressions
- `benchmark run`
  - executes end-to-end synthesis + analysis benchmark pass
  - emits objective/proxy/speed metrics
- `benchmark check`
  - compares benchmark report against baseline thresholds

## Dataset Schema

`benchmark run` expects `latent-ir.benchmark.dataset.v1`:

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

`id` is required. Better targets produce more useful regressions.

## Run Examples

```bash
cargo run -- eval text \
  --dataset datasets/eval_text.json \
  --model models/text_encoder_v1.json \
  --output reports/eval_text.json

cargo run -- benchmark run \
  --dataset datasets/benchmark_suite.json \
  --text-model models/text_encoder_v1.json \
  --audio-model models/audio_encoder_v1.json \
  --repeats 3 \
  --output reports/benchmark.json
```

## Regression Gate Examples

```bash
cargo run -- eval check \
  --report reports/eval_text.json \
  --baseline ci/baselines/eval_text_baseline.json \
  --max-regression 0.10

cargo run -- benchmark check \
  --report reports/benchmark.json \
  --baseline ci/baselines/benchmark_baseline.json \
  --max-regression 0.05
```

## Spatial Regression Coverage

Dataset:
- `ci/datasets/spatial_corpus_ci.json`

Harness:
- `tests/spatial_corpus_tests.rs`

Coverage includes:
- 7.2.4 beds
- custom 16-channel rings
- custom A.B.C-style elevated layouts
- cartesian-only custom ingestion

## Render QA Coverage

Harness:
- `tests/render_spatial_qa_tests.rs`

Checks include:
- streaming vs direct parity
- output-length correctness
- block-boundary integrity

## CI Entry Point

```bash
./scripts/ci_regression_gates.sh
```

This mirrors `.github/workflows/regression-gates.yml`.

## Operational Recommendation

For release candidates:
1. run full gates locally
2. review warning deltas (not only pass/fail)
3. refresh baselines only when behavior changes are intentional and documented
