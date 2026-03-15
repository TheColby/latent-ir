#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p ci/out

echo "[ci] generating eval report..."
cargo run --quiet -- eval text \
  --dataset ci/datasets/eval_text_ci.json \
  --model examples/models/text_encoder_v1.json \
  --sample-rate 48000 \
  --seed 1234 \
  --output ci/out/eval_text_report.json

echo "[ci] checking eval report against baseline..."
cargo run --quiet -- eval check \
  --report ci/out/eval_text_report.json \
  --baseline ci/baselines/eval_text_baseline.json \
  --max-regression 0.10

echo "[ci] generating benchmark report..."
cargo run --quiet -- benchmark run \
  --dataset ci/datasets/benchmark_ci.json \
  --text-model examples/models/text_encoder_v1.json \
  --sample-rate 48000 \
  --seed 2026 \
  --repeats 2 \
  --output ci/out/benchmark_report.json

echo "[ci] checking benchmark report against baseline..."
cargo run --quiet -- benchmark check \
  --report ci/out/benchmark_report.json \
  --baseline ci/baselines/benchmark_baseline.json \
  --max-regression 0.10

echo "[ci] running spatial corpus metric envelopes..."
cargo test --quiet --test spatial_corpus_tests spatial_corpus_metrics_stay_within_envelopes

echo "[ci] running streaming spatial render QA..."
cargo test --quiet --test render_spatial_qa_tests

echo "[ci] running dataset integrity/leakage verification smoke..."
VERIFY_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/latent_ir_dataset_verify.XXXXXX")"
trap 'rm -rf "$VERIFY_TMP_DIR"' EXIT

cargo run --quiet -- dataset synthesize \
  --out-dir "$VERIFY_TMP_DIR" \
  --count 12 \
  --seed 606 \
  --sample-rate 8000 \
  --channels mono \
  --duration-min 0.10 --duration-max 0.14 \
  --t60-min 0.20 --t60-max 0.45 \
  --predelay-max-ms 8

cargo run --quiet -- dataset split \
  --manifest "$VERIFY_TMP_DIR/manifest.dataset.json" \
  --output "$VERIFY_TMP_DIR/split.dataset.json" \
  --seed 606 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --lock-hashes

cargo run --quiet -- dataset verify \
  --split-manifest "$VERIFY_TMP_DIR/split.dataset.json"

echo "[ci] regression gates passed"
