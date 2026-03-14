# latent-ir Roadmap

## v0.1 Foundation (completed)

- CLI with core commands: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`
- Canonical descriptor model + presets
- Procedural deterministic IR synthesis
- Engineering analysis reports + JSON outputs
- Baseline offline rendering pipeline
- Initial test coverage for deterministic behavior

## v0.2 Acoustics + Data Model (mostly completed)

- Improved reflection/tail shaping and decay robustness
- Reference-audio descriptor extraction baseline
- Metadata/report schema versioning
- Learned JSON text/audio encoders
- Optional ONNX inference backend (feature-gated)
- `train-encoder`, `eval`, and `benchmark` workflows
- Perceptual macro controls + trajectory automation
- FFT-partitioned render engine

## v0.3 Hybrid Conditioning + Spatial Rigor (in progress)

Delivered in this phase so far:

- Model manifest validation (`model validate`)
- CI regression gates (`eval check`, `benchmark check`)
- Streaming FFT render mode (`render --engine fft-streaming`)
- Custom layout ingestion with polar/cartesian validation
- Spatial channel-map sidecar emission/ingestion
- Spatial corpus envelope regression tests
- Spatial render QA (streaming/direct parity + block-boundary checks)
- Geometry-aware custom synthesis from `position_m`
  - distance delay/gain/HF shaping
  - image-source-lite early reflections
- Virtual source/listener controls
- Arrival spread + ITD/IACC-style analysis metrics

Remaining high-impact items for v0.3:

- Latent-vector projection contracts and constrained priors
- Conditioning confidence/uncertainty reporting
- Descriptor trajectory-coupled morphing

## v1.0 Research/Product Grade

- Dataset-scale batch tooling and reproducibility hardening
- Higher-performance render kernels and memory optimization
- Broader validation against external references
- Stable library APIs while preserving CLI-first workflow
- Extended docs for model development and evaluation lifecycle

## Next Milestone Focus

1. Latent-ready conditioning contracts (`LatentVector` + constrained projection APIs)
2. Confidence/uncertainty propagation through conditioning and metadata
3. Descriptor-trajectory-conditioned morphing and generation controls
4. Larger spatial benchmark suite and trend dashboards
