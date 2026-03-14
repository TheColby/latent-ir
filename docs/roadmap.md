# latent-ir Roadmap

## v0.1 Foundation (completed)

- CLI core: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`
- canonical descriptor model + presets
- deterministic procedural IR synthesis
- engineering analysis reports + JSON outputs
- baseline offline rendering pipeline
- initial deterministic/regression tests

## v0.2 Acoustics + Data Model (mostly completed)

- improved reflection/tail shaping
- reference-audio descriptor extraction baseline
- metadata/report schema versioning
- learned JSON text/audio encoders
- optional ONNX inference backend
- `train-encoder`, `eval`, `benchmark` workflows
- perceptual macro controls + trajectory automation
- FFT-partitioned render engine

## v0.3 Hybrid Conditioning + Spatial Rigor (in progress)

Delivered so far:

- model manifest validation (`model validate`)
- CI regression gates (`eval check`, `benchmark check`)
- streaming FFT render mode (`render --engine fft-streaming`)
- workload-aware render auto-selection
- custom layout ingestion with polar/cartesian validation
- spatial channel-map sidecar emission/ingestion
- spatial corpus envelope regression tests
- spatial render QA (streaming/direct parity + boundary checks)
- geometry-aware custom synthesis from `position_m`
  - distance delay/gain/HF shaping
  - image-source-lite early reflections
- virtual source/listener controls
- arrival spread + ITD/IACC-style metrics
- generation guardrails for tail preservation
- optional sample-rate auto-resampling for `render` and `morph`
- selectable resample modes (`linear` / `cubic`) for reconciliation paths
- expanded CLI warning surfacing and validation hints
- analysis confidence metrics and replay-command metadata for reproducibility/transparency

Remaining high-impact items for v0.3:

- latent-vector projection contracts and constrained priors
- conditioning confidence/uncertainty reporting
- descriptor trajectory-coupled morphing

## v1.0 Research/Product Grade

- dataset-scale batch tooling + reproducibility hardening
- higher-performance render kernels and memory optimization
- broader validation against external references
- stable library APIs while preserving CLI-first workflow
- expanded model-development/evaluation lifecycle docs

## Next Focus (Recommended)

1. Latent-ready conditioning contracts (`LatentVector` APIs)
2. Confidence/uncertainty propagation in metadata and eval outputs
3. Trajectory-conditioned morph/generation controls
4. Larger spatial benchmark suites + trend dashboards
