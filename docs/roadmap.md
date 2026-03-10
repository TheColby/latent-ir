# latent-ir Roadmap

## v0.1 - Foundation (completed)

- Stable CLI with subcommands: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`
- Canonical descriptor model and preset system
- Procedural deterministic IR generator (mono/stereo + FOA/surround/Atmos layouts)
- Approximate engineering analysis metrics and JSON reports
- Offline convolution rendering path
- Core tests for deterministic behavior and pipeline sanity

## v0.2 - Better Acoustics + Data Model (mostly completed)

- Improved early-reflection statistical models
- Better band decay estimation and robustness checks
- Reference-audio descriptor extraction (non-learned baseline)
- IR metadata schema versioning
- Batch generation/analyze modes and richer machine-readable outputs
- Learned JSON text/audio encoders
- Optional ONNX inference backend (feature-gated)
- `train-encoder` and `eval` command workflows
- Benchmark lab with CI-friendly `benchmark check`
- Perceptual macro controls + trajectory automation
- FFT-partitioned render engine with direct-path reference tests

## v0.3 - Hybrid Conditioning Interfaces (in progress)

- Explicit conditioning traits for pluggable learned modules
- Optional text encoder integration behind feature flags
- Optional reference-audio embedding adapter
- Descriptor constraint projection and confidence tracking
- Morphing upgrades (envelope-aware + descriptor-space coupling)
- Model manifest schema + runtime compatibility validation (`model validate`)
- Custom channel-layout ingestion (`--channels custom --layout-json ...`)
- Validated channel-map sidecar emission/ingestion (`*.channels.json`)
- Upgraded FFT partitioned convolution core (FDL overlap-save style)
- Multichannel analysis metrics (correlation matrix + directional energy balances)

Remaining for v0.3:

- Latent-space descriptor priors and constrained projection APIs
- Confidence calibration/uncertainty reporting in conditioning chain
- Descriptor trajectory-conditioned morphing coupling

## v1.0 - Research/Product Grade Toolkit

- Robust reproducible pipelines for dataset-scale runs
- Optimized rendering paths (FFT partitioned convolution)
- Stronger acoustic metric suite and validation harness
- Plugin-ready library APIs (while staying CLI-first)
- End-to-end docs for hybrid DSP + ML model development and evaluation

## Next Milestone Focus

1. Latent-ready conditioning contracts (`LatentVector` + projection interfaces)
2. Benchmark/eval baseline suites in CI for mandatory regression gates
3. Layout-aware render/analyze benchmark corpus for large custom arrays
