<p align="center">
  <img src="docs/assets/latent-ir-logo.svg" alt="latent-ir logo" width="760" />
</p>

# latent-ir

**Generative impulse responses and acoustic spaces for the command line.**

`latent-ir` is a CLI-first Rust toolkit for generating, analyzing, morphing, and rendering impulse responses (IRs) with a hybrid DSP + ML architecture.

Core contract:
- conditioning layers infer intent
- DSP synthesis produces and validates the final IR

Current scope is explicit: production-grade procedural DSP plus rule-based/lightweight learned conditioning hooks, not a large pretrained reverb foundation model.

## Thesis

Most reverb workflows are static: fixed presets, fixed measured IR libraries, or plugin internals that are difficult to inspect.

`latent-ir` treats reverberation as a controllable acoustic design space:
- descriptor-conditioned
- deterministic and reproducible
- machine-readable and scriptable
- extensible toward richer learned modules

## Why This Exists

- Give audio DSP engineers a serious command-line acoustics lab.
- Give ML engineers a deterministic synthesis backbone for conditioning research.
- Give sound designers a reproducible alternative to fixed-space libraries.

## Feature Snapshot

- Commands: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`, `dataset`
- Learned tooling: `train-encoder`, `eval`, `benchmark`, `model validate`, `ab-test`
- AI research tooling: `dataset synth` / `dataset split` / `dataset verify` for reproducible corpus generation, hash-locked train/val/test splits, and leakage/integrity checks
- Canonical descriptor model (`DescriptorSet`) across time/spectral/structural/spatial domains
- Deterministic procedural IR generation with seed control
- Tail-protection guardrail in `generate` (opt out with `--allow-tail-truncation`)
- Optional explicit file-end taper in `generate` (`--tail-fade-ms`) to force smooth decay to zero
- Prompt parser extracts more explicit acoustic directives (for example RT60, predelay, duration, channel-hint tokens)
- Spatial support:
  - built-ins: `mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`
  - custom layouts via JSON (`--channels custom --layout-json ...`)
- Geometry-aware synthesis with custom `position_m` layouts:
  - distance-based delay/gain/HF shaping
  - image-source-lite early reflection clusters
  - virtual source/listener controls
- Analysis metrics (console + JSON):
  - EDT / T20 / T30 / T60 engineering estimates
  - decay-span and confidence summaries for decay metrics
  - tail reach and tail margin diagnostics (`tail_reaches_minus60db_s`, `tail_margin_to_end_s`)
  - crest factor summary for transient/energy balance context
  - predelay, spectral centroid, low/mid/high decay summaries
  - inter-channel correlation + ITD/IACC-style coherence summaries
  - directional energy summaries with channel map
- Quality gates for launch/release workflows (`--quality-gate --quality-profile lenient|launch|strict`)
- Conditioning uncertainty estimates in metadata (`conditioning.uncertainty.*`) and eval outputs (`uncertainty_metrics.*`)
- Render engines: `direct`, `fft-partitioned`, `fft-streaming`
- `render` and `morph` optional sample-rate reconciliation via `--auto-resample --resample-mode linear|cubic`
- Trajectory-conditioned morphing via `morph --alpha-trajectory <json>`
- Benchmark trend dashboards via `benchmark trend --reports ... --output trend.md`
- Reproducible sidecars: metadata JSON, analysis JSON, channel map JSON

## Practical Guardrails

- Descriptor clamp changes are surfaced as warnings.
- Tail-truncation risk is surfaced and auto-corrected by default.
- Optional end taper can enforce exact zero at file end (`--tail-fade-ms`).
- Mixed sample-rate workflows get explicit guidance or optional auto-resampling.
- Analysis caveats are always printed in console output.
- Metadata now includes a replay command string and combined conditioning delta.
- Metadata now includes reproducibility fingerprints (`ir_sha256`, `descriptor_sha256`, `channel_map_sha256`).
- Optional quality gate can fail fast in CI/release pipelines while still writing analysis artifacts.

## Installation

Prerequisite: Rust stable (`rustup`, `cargo`).

Install from source:

```bash
cargo install --path . --locked
```

Verify:

```bash
latent-ir --help
```

If binary is not found:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Or use helper:

```bash
./scripts/install_local.sh
```

## Build and Test

```bash
cargo fmt
cargo test
```

CI-equivalent local gates:

```bash
./scripts/ci_regression_gates.sh
```

## Quickstart

Generate from prompt:

```bash
cargo run -- generate \
  --prompt "vast icy cathedral" \
  --explain-conditioning \
  --t60 12 \
  --tail-fade-ms 35 \
  --channels stereo \
  --seed 1337 \
  --output out/cathedral_ir.wav
```

Analyze:

```bash
cargo run -- analyze out/cathedral_ir.wav --json
```

Gate analysis quality for release:

```bash
cargo run -- analyze out/cathedral_ir.wav \
  --quality-gate --quality-profile launch
```

Morph:

```bash
cargo run -- morph cave_ir.wav plate_ir.wav --alpha 0.4 --output out/morphed.wav
```

Trajectory-conditioned morph:

```bash
cargo run -- morph cave_ir.wav plate_ir.wav \
  --alpha-trajectory examples/morph/alpha_ramp.json \
  --output out/morphed_traj.wav
```

Render:

```bash
cargo run -- render dry_input.wav --ir out/morphed.wav --mix 0.25 --output out/rendered.wav
```

Dataset synthesis for encoder research / augmentation:

```bash
cargo run -- dataset synthesize \
  --out-dir out/research_dataset \
  --count 256 \
  --channels stereo \
  --quality-gate --quality-profile launch \
  --export-training-json
```

Create deterministic, hash-locked train/val/test splits:

```bash
cargo run -- dataset split \
  --manifest out/research_dataset/manifest.dataset.json \
  --output out/research_dataset/split.dataset.json \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --lock-hashes --emit-training-json
```

Verify split integrity and leakage constraints:

```bash
cargo run -- dataset verify \
  --split-manifest out/research_dataset/split.dataset.json \
  --fail-on-prompt-overlap \
  --output out/research_dataset/verify.dataset.json
```

## Spatial Examples

FOA (ambiX):

```bash
cargo run -- generate \
  --prompt "massive concrete grain silo" \
  --channels foa \
  --output out/silo_foa.wav
```

7.2.4 bed:

```bash
cargo run -- generate \
  --prompt "industrial aircraft hangar" \
  --channels 7.2.4 \
  --output out/hangar_724.wav
```

Custom 16-channel cartesian layout + geometry steering:

```bash
cargo run -- generate \
  --prompt "32-second dark cave" \
  --duration 32 \
  --sample-rate 384000 \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_circle_10m_origin_cartesian_only.json \
  --source-x-m 0.0 --source-y-m 24.0 --source-z-m 2.0 \
  --listener-x-m 0.0 --listener-y-m 0.0 --listener-z-m 1.5 \
  --output out/dark_cave_16ch_384k.wav
```

## Render Performance Tips

- Use `--engine auto` unless you have a reason not to.
- Prefer `fft-streaming` for long multichannel material.
- `--partition-size 2048` or `4096` is usually a good baseline for FFT paths.

## Output Artifacts

`generate` can emit:
- WAV IR
- metadata JSON (`latent-ir.generation.v1`)
- analysis JSON (`latent-ir.analysis.v1`)
- channel-map JSON (`latent-ir.channel-map.v1`)

Metadata includes:
- project/version
- command/prompt/preset/seed/sample-rate
- resolved descriptor values
- conditioning trace
- reproducibility fingerprints (`ir_sha256`, `descriptor_sha256`, `channel_map_sha256`)
- warnings
- embedded analysis summary

## Command Notes

- `generate`
  - supports prompt/preset/overrides/learned models/custom layout/geometry control
  - validates and reports descriptor corrections
  - if `--channels` is omitted, prompt/preset channel-format intent can resolve output format
  - `--explain-conditioning` prints resolved conditioning deltas and descriptor snapshot
  - `--tail-fade-ms` applies an explicit end taper so output reaches exact zero at file end
  - optional quality gate (`--quality-gate --quality-profile ...`) with pass/fail checks in metadata
- `analyze`
  - prints metrics, confidence estimates, and warnings; supports JSON output
  - optional quality gate (`--quality-gate --quality-profile ...`) for release validation
- `morph`
  - supports `--auto-resample --resample-mode linear|cubic` when IR sample rates differ
  - supports `--alpha-trajectory` for time-varying alpha interpolation
- `render`
  - supports `--auto-resample --resample-mode linear|cubic` and workload-aware engine selection
- `dataset synthesize`
  - generates IR corpora (WAV + metadata + analysis + channel maps)
  - supports prompt-bank control, descriptor jitter, preset mixing, and optional quality gating
  - can export `training_text.json` and `training_audio.json` compatible with `train-encoder`
- `dataset split`
  - creates deterministic train/val/test manifests with configurable ratios
  - optional hash-locking reads per-sample metadata hashes into split records
  - optional split-specific train-encoder JSON exports
- `dataset verify`
  - validates split integrity (missing files, hash mismatches, split ID overlap)
  - can fail on prompt overlap (`--fail-on-prompt-overlap`) for stricter anti-leakage policy
  - writes machine-readable verification report (`latent-ir.dataset-verify.v1`)
- `benchmark trend`
  - builds Markdown + JSON trend dashboards from multiple benchmark reports
- `eval text` / `eval audio`
  - now emit uncertainty summaries (`mean_confidence`, `mean_uncertainty`)

## Top 5 Likely Complaints (And What Is Implemented)

1. “Prompt interpretation is too vague.”
- Numeric prompt parsing now handles explicit RT60, predelay, duration, and channel-format hints.

2. “You report RT values but don’t show truncation risk.”
- Analysis now emits tail diagnostics (`tail_reaches_minus60db_s`, `tail_margin_to_end_s`) and warnings when tails are likely clipped.

3. “I can’t reproduce or verify artifacts exactly.”
- Metadata includes replay command, full conditioning trace, and reproducibility hashes for IR/descriptor/channel-map payloads.

4. “Split leakage and dataset integrity aren’t enforced.”
- `dataset verify` now checks split IDs, prompt overlap, missing artifacts, and hash-lock mismatches (`--fail-on-prompt-overlap` for strict mode).

5. “There is no machine-checkable quality bar before release.”
- `generate` and `analyze` support quality gates with `lenient`, `launch`, and `strict` profiles and non-zero exit on failure.

## Demo Pack

Generate deterministic launch/demo assets:

```bash
./scripts/generate_demo_pack.sh
```

## Limitations and Honesty

- v0.x is DSP-first with rule-based and lightweight learned conditioning.
- Current metrics are engineering estimates for reproducible workflow comparisons.
- Geometry-driven reflections are image-source-lite approximations.
- This is not a full geometric/wave simulator or an object-based Atmos renderer.

## Documentation Index

- [docs/architecture.md](docs/architecture.md)
- [docs/spatial-layouts.md](docs/spatial-layouts.md)
- [docs/perceptual-controls.md](docs/perceptual-controls.md)
- [docs/learned-encoders.md](docs/learned-encoders.md)
- [docs/model-manifests.md](docs/model-manifests.md)
- [docs/benchmarking.md](docs/benchmarking.md)
- [docs/datasets.md](docs/datasets.md)
- [docs/launch-readiness.md](docs/launch-readiness.md)
- [docs/roadmap.md](docs/roadmap.md)

## Contributing

Contributions are welcome.

Recommended workflow:
1. Open an issue with intended behavior and constraints.
2. Add/update tests for DSP behavior, schemas, and CLI UX.
3. Keep claims precise and technically defensible.
4. Run `cargo fmt`, `cargo test`, and `./scripts/ci_regression_gates.sh`.

## Attribution

latent-ir is developed by Colby Leider and the open-source contributor community.

## License

MIT. See [LICENSE](LICENSE).
