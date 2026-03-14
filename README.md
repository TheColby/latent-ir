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

- Commands: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`
- Learned tooling: `train-encoder`, `eval`, `benchmark`, `model validate`, `ab-test`
- Canonical descriptor model (`DescriptorSet`) across time/spectral/structural/spatial domains
- Deterministic procedural IR generation with seed control
- Tail-protection guardrail in `generate` (opt out with `--allow-tail-truncation`)
- Spatial support:
  - built-ins: `mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`
  - custom layouts via JSON (`--channels custom --layout-json ...`)
- Geometry-aware synthesis with custom `position_m` layouts:
  - distance-based delay/gain/HF shaping
  - image-source-lite early reflection clusters
  - virtual source/listener controls
- Analysis metrics (console + JSON):
  - EDT / T20 / T30 / T60 engineering estimates
  - predelay, spectral centroid, low/mid/high decay summaries
  - inter-channel correlation + ITD/IACC-style coherence summaries
  - directional energy summaries with channel map
- Render engines: `direct`, `fft-partitioned`, `fft-streaming`
- `render` and `morph` optional sample-rate reconciliation via `--auto-resample`
- Reproducible sidecars: metadata JSON, analysis JSON, channel map JSON

## Practical Guardrails

- Descriptor clamp changes are surfaced as warnings.
- Tail-truncation risk is surfaced and auto-corrected by default.
- Mixed sample-rate workflows get explicit guidance or optional auto-resampling.
- Analysis caveats are always printed in console output.

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
  --t60 12 \
  --channels stereo \
  --seed 1337 \
  --output out/cathedral_ir.wav
```

Analyze:

```bash
cargo run -- analyze out/cathedral_ir.wav --json
```

Morph:

```bash
cargo run -- morph cave_ir.wav plate_ir.wav --alpha 0.4 --output out/morphed.wav
```

Render:

```bash
cargo run -- render dry_input.wav --ir out/morphed.wav --mix 0.25 --output out/rendered.wav
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
- warnings
- embedded analysis summary

## Command Notes

- `generate`
  - supports prompt/preset/overrides/learned models/custom layout/geometry control
  - validates and reports descriptor corrections
- `analyze`
  - prints metrics and warnings, supports JSON output
- `morph`
  - supports `--auto-resample` when IR sample rates differ
- `render`
  - supports `--auto-resample` and workload-aware engine selection

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
