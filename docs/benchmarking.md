# Benchmarking Lab

`latent-ir` includes a benchmark/evaluation lab designed for regression tracking and CI gating.

## Dataset schema

Benchmark dataset JSON:

```json
{
  "schema_version": "latent-ir.benchmark.dataset.v1",
  "samples": [
    {
      "id": "sample_001",
      "prompt": "dark cathedral",
      "reference_audio": "refs/hall_ir.wav",
      "target_ir": "targets/hall_target.wav",
      "target_descriptor": { "... DescriptorSet ...": "..." }
    }
  ]
}
```

Fields are optional except `id`; best results come from supplying both `target_ir` and `target_descriptor`.

## Running benchmark

```bash
cargo run -- benchmark run \
  --dataset datasets/benchmark_suite.json \
  --text-model models/text_encoder_v1.json \
  --audio-model models/audio_encoder_v1.json \
  --repeats 3 \
  --output reports/benchmark.json
```

Report schema: `latent-ir.benchmark.v1`

Report sections:

- `objective`: descriptor + analysis error
- `speed`: encode/generate/analyze/total latency
- `stability`: repeat-run stddev summaries
- `perceptual_proxy`: clarity/brightness/spaciousness/distance proxy errors
- `summary`: aggregate scores for regression checks

## CI-style gating

```bash
cargo run -- benchmark check \
  --report reports/benchmark_new.json \
  --baseline reports/benchmark_baseline.json \
  --max-regression 0.05
```

The command fails when selected key metrics regress beyond threshold.
