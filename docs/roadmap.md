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
- analysis tail diagnostics (`tail_reaches_minus60db_s`, `tail_margin_to_end_s`) and crest-factor reporting
- reproducibility hashes in generation metadata (`ir_sha256`, `descriptor_sha256`, `channel_map_sha256`)
- quality-gate profiles for `generate`/`analyze` (`lenient`, `launch`, `strict`)
- `dataset synthesize` command for AI corpus generation + augmentation controls
  - prompt-bank sampling
  - preset mixing + descriptor jitter
  - optional training JSON exports (`training_text.json`, `training_audio.json`)
  - dataset manifest (`latent-ir.dataset.v1`)
- dataset split/version tooling
  - deterministic train/val/test split manifests (`latent-ir.dataset-split.v1`)
  - optional hash-locking with per-sample metadata hashes
  - optional split-specific train-encoder JSON exports
- conditioning confidence/uncertainty propagation
  - generation metadata includes conditioning uncertainty summaries
  - eval reports include uncertainty metrics (`mean_confidence`, `mean_uncertainty`)
- trajectory-conditioned morph controls
  - `morph --alpha-trajectory` supports time-varying blend envelopes
- benchmark trend dashboards + spatial benchmark expansion
  - `benchmark trend` emits Markdown + JSON trend artifacts
  - spatial corpus now includes FOA and 7.1 envelope coverage

Remaining high-impact items for v0.3:

- latent-vector projection contracts and constrained priors
- trajectory-conditioned generation beyond macro control

## v1.0 Research/Product Grade

- dataset-scale batch tooling + reproducibility hardening
- higher-performance render kernels and memory optimization
- broader validation against external references
- stable library APIs while preserving CLI-first workflow
- expanded model-development/evaluation lifecycle docs

## Next Focus (Recommended)

1. Latent-ready conditioning contracts (`LatentVector` APIs)
2. Dataset curation tooling (`dedupe`, stratification, provenance audit)
3. Trajectory-conditioned generation beyond macro trajectories
4. External metrology cross-validation for analysis metrics
5. Render-kernel performance optimization for very large multichannel jobs
