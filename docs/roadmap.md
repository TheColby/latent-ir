# latent-ir Roadmap

## v0.1 - Foundation (current scaffold)

- Stable CLI with subcommands: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`
- Canonical descriptor model and preset system
- Procedural deterministic IR generator (mono/stereo)
- Approximate engineering analysis metrics and JSON reports
- Offline convolution rendering path
- Core tests for deterministic behavior and pipeline sanity

## v0.2 - Better Acoustics + Data Model

- Improved early-reflection statistical models
- Better band decay estimation and robustness checks
- Reference-audio descriptor extraction (non-learned baseline)
- IR metadata schema versioning
- Batch generation/analyze modes and richer machine-readable outputs

## v0.3 - Hybrid Conditioning Interfaces

- Explicit conditioning traits for pluggable learned modules
- Optional text encoder integration behind feature flags
- Optional reference-audio embedding adapter
- Descriptor constraint projection and confidence tracking
- Morphing upgrades (envelope-aware + descriptor-space coupling)

## v1.0 - Research/Product Grade Toolkit

- Robust reproducible pipelines for dataset-scale runs
- Optimized rendering paths (FFT partitioned convolution)
- Stronger acoustic metric suite and validation harness
- Plugin-ready library APIs (while staying CLI-first)
- End-to-end docs for hybrid DSP + ML model development and evaluation
