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

echo "[ci] regression gates passed"
