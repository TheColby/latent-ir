<p align="center">
  <img src="docs/assets/latent-ir-logo.svg" alt="latent-ir logo" width="760" />
</p>

# latent-ir

**Generative impulse responses and acoustic spaces for the command line.**

`latent-ir` is a CLI-first Rust toolkit for generating, analyzing, morphing, and rendering impulse responses (IRs) with a hybrid DSP+ML architecture:

- ML-facing conditioning resolves acoustic intent.
- DSP synthesis constructs, constrains, and validates the final IR.

This repository is intentionally honest about scope: current releases are production-grade procedural DSP with lightweight learned/rule-based conditioning hooks, not a foundation-model reverb engine.

## Thesis

Most reverb workflows are static: fixed presets, fixed measured IR libraries, or opaque plugin internals.

`latent-ir` treats reverberation as a **controllable acoustic design space**:

- descriptor-conditioned
- scriptable and reproducible
- analyzable at scale
- extensible toward learned modules

## Why This Repo Exists

- Give audio DSP engineers a serious CLI lab for synthetic IR design.
- Give ML engineers a deterministic DSP backbone for hybrid conditioning research.
- Give sound designers a programmable alternative to fixed-space libraries.

## Feature Overview

- CLI commands: `generate`, `analyze`, `morph`, `render`, `sample`, `preset`
- Learned tooling: `train-encoder`, `eval`, `benchmark`, `model validate`, `ab-test`
- Canonical descriptor model (`DescriptorSet`) across time/spectral/structural/spatial domains
- Deterministic procedural IR generation with seed control
- Tail-preserving duration floor in `generate` (opt out with `--allow-tail-truncation`)
- Spatial layout support:
  - built-ins: `mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`
  - custom JSON layouts (`--channels custom --layout-json ...`)
- Cartesian geometry support for custom layouts (`position_m`) with polar/cartesian consistency checks
- Geometry-aware synthesis:
  - distance-based delay
  - distance gain falloff
  - HF air-loss shaping
  - image-source-lite early reflections
- Virtual source/listener controls:
  - `--source-x-m --source-y-m --source-z-m`
  - `--listener-x-m --listener-y-m --listener-z-m`
- Analysis reports (console + JSON):
  - EDT / T20 / T30 / T60 engineering estimates
  - predelay estimate, spectral centroid, low/mid/high decay summaries
  - inter-channel correlation matrix + summaries
  - directional energy (front/rear/height/LFE with channel map)
  - arrival spread metrics
  - ITD-ish and IACC-style coherence summaries
- Rendering engines:
  - `direct`
  - `fft-partitioned`
  - `fft-streaming` (long-form multichannel)
- Optional sample-rate reconciliation for `render` and `morph` with `--auto-resample`
- Reproducible metadata sidecars and channel-map sidecars
- CI regression gates for eval/benchmark/spatial/render QA

## Architecture Overview

Pipeline:

```text
prompt / reference audio / explicit params
-> conditioning layer
-> DescriptorSet (canonical acoustic state)
-> ProceduralIrGenerator
-> validation + normalization
-> final IR
-> IrAnalyzer + machine-readable metadata
```

Primary modules:

- `src/core/descriptors`: canonical acoustic model
- `src/core/semantics`: rule-based prompt conditioning
- `src/core/conditioning`: learned conditioning interfaces + model adapters
- `src/core/presets`: preset descriptors
- `src/core/generator`: DSP IR synthesis
- `src/core/analysis`: engineering acoustic metrics
- `src/core/spatial`: custom layout parsing/validation/channel maps
- `src/core/morph`: IR + descriptor interpolation
- `src/core/render`: offline convolution paths

## Launch-Ready Answers (Common Questions)

### 1) "Where are the audio demos?"

Generate a reproducible local demo pack:

```bash
./scripts/generate_demo_pack.sh
```

This writes deterministic IR assets and analysis JSON to `out/demos/`, suitable for A/B posting with your own dry material.

### 2) "Is this really ML, or mostly DSP + rules?"

Current status is explicit:

- production path: procedural DSP synthesis
- conditioning: presets + semantic rules + optional lightweight learned encoders
- future path: richer learned models behind stable descriptor interfaces

No hidden "black-box AI" claims in core generation.

### 3) "How accurate are your RT/EDT metrics?"

Metrics are labeled and documented as **engineering estimates** for v0.
They are deterministic and useful for workflow consistency, but are not standards-certified architectural acoustics metrology.

### 4) "What exactly is implemented for spatial audio?"

Supported layout modes and semantics are explicit:

| Mode | Status | Notes |
|---|---|---|
| `mono`, `stereo` | implemented | standard IR generation/analysis/render |
| `foa` (ambiX) | implemented | channel map + analysis integration |
| `5.1`, `7.1`, `7.1.4`, `7.2.4` | implemented | bed-style channels, directional metrics via map |
| `custom` | implemented | JSON layout with `position_m` and/or az/el |
| object-based Atmos renderer | not implemented | non-goal in current scope |

For custom arrays, `position_m` is authoritative when provided.

### 5) "How does this fit a real production workflow?"

Typical loop:

1. `generate` IR variants from prompts/presets/overrides.
2. `analyze` and filter with JSON metrics.
3. `morph` candidate IRs.
4. `render` dry stems offline (direct/FFT/streaming).
5. audition in DAW or convolution host.

### 6) "Can this scale for long multichannel jobs?"

Yes, via render engines and auto-selection:

- small jobs: `direct`
- medium jobs: `fft-partitioned`
- large jobs: `fft-streaming` (block-wise, reduced memory pressure)
- `render --engine auto` now chooses engine based on workload size and reports the selected mode.

### 7) "Install friction / PATH issues"

Use install helper:

```bash
./scripts/install_local.sh
```

If needed, add cargo bin path:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Current Status

Implemented:

- serious DSP-first IR generator (deterministic, seeded)
- spatial layouts including custom geometry
- image-source-lite early reflection augmentation for custom geometry paths
- rich analysis metrics and JSON output schema
- render paths including streaming FFT for multichannel long-form use
- learned encoder training/eval baseline workflow
- CI regression gates

Not implemented (by design in current stage):

- full geometric acoustics simulation
- standards-certified room-acoustics metrology
- large pretrained transformer/diffusion IR generators in core path

## Installation

Prerequisite: Rust stable toolchain (`rustup`, `cargo`).

Install from source tree:

```bash
cargo install --path . --locked
```

Verify:

```bash
latent-ir --help
```

If `latent-ir` is not found, ensure cargo bin is on `PATH` (`~/.cargo/bin`).

## Build and Test

```bash
cargo fmt
cargo test
```

## CI Regression Gates

Run the same gates locally that CI uses:

```bash
./scripts/ci_regression_gates.sh
```

This runs:

- `eval check` vs committed baseline
- `benchmark check` vs committed baseline
- spatial corpus envelope tests
- streaming spatial render QA tests

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

Analyze IR:

```bash
cargo run -- analyze out/cathedral_ir.wav --json
```

Morph IRs:

```bash
cargo run -- morph cave_ir.wav plate_ir.wav --alpha 0.4 --output out/morphed.wav
```

Render dry audio through IR:

```bash
cargo run -- render dry_input.wav --ir out/morphed.wav --mix 0.25 --output out/rendered.wav
```

## Spatial Examples

Generate FOA (ambiX):

```bash
cargo run -- generate \
  --prompt "massive concrete grain silo" \
  --channels foa \
  --output out/silo_foa.wav
```

Generate 7.2.4 bed:

```bash
cargo run -- generate \
  --prompt "industrial aircraft hangar" \
  --channels 7.2.4 \
  --output out/hangar_724.wav
```

Generate custom 16-channel from polar layout JSON:

```bash
cargo run -- generate \
  --prompt "circular concrete tank with long smooth tail" \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_ring.json \
  --output out/tank_16.wav
```

Generate custom 16-channel from cartesian-only layout JSON:

```bash
cargo run -- generate \
  --prompt "32-second dark cave" \
  --duration 32 \
  --sample-rate 384000 \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_circle_10m_origin_cartesian_only.json \
  --output out/dark_cave_16ch_cart_384k.wav
```

Geometry-steered generation (virtual source/listener):

```bash
cargo run -- generate \
  --prompt "aircraft hangar long metallic tail" \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_circle_10m_origin_cartesian_only.json \
  --source-x-m 0.0 --source-y-m 24.0 --source-z-m 2.0 \
  --listener-x-m 0.0 --listener-y-m 0.0 --listener-z-m 1.5 \
  --output out/hangar_geo_16.wav
```

Render long-form content with streaming FFT:

```bash
cargo run -- render long_program.wav \
  --ir out/hangar_geo_16.wav \
  --engine fft-streaming \
  --partition-size 4096 \
  --mix 0.25 \
  --output out/rendered_streaming.wav
```

## Command Reference

- `generate`
  - Inputs: prompt/preset, learned models (JSON or optional ONNX), descriptor overrides, channel layout options, geometry controls, seed, sample rate
  - Outputs: WAV IR, metadata JSON, channel-map JSON, optional analysis JSON
  - Console: detailed metrics (decay, spectral, spatial/correlation, arrival/ITD/IACC) + clamp/auto-adjust warnings
- `analyze`
  - Inputs: IR WAV, optional channel map
  - Outputs: console metrics and/or JSON report
- `morph`
  - Inputs: two IR WAVs and `--alpha`
  - Output: morphed IR WAV
  - Optional `--auto-resample` when source IR sample rates differ
- `render`
  - Inputs: dry WAV, IR WAV, `--mix`
  - Engines: `auto|direct|fft-partitioned|fft-streaming`
  - Optional `--auto-resample` when dry and IR sample rates differ
- `sample`
  - Outputs random descriptors (text/JSON)
- `preset`
  - Lists preset names or prints preset descriptors
- `train-encoder`
  - `text` and `audio` model fitting utilities
- `eval`
  - `text`, `audio`, and `check` modes
- `benchmark`
  - `run` and `check` modes
- `model`
  - `validate` model manifests
- `ab-test`
  - one-shot comparison harness (industrial model vs baseline)

## Presets and Prompt Ideas

Built-in presets:

- `intimate_wood_chapel`
- `dark_stone_cathedral`
- `steel_bunker`
- `glass_corridor`
- `frozen_plate`
- `impossible_infinite_tunnel`

Prompt ideas (free-form text):

- `vast icy cathedral with long tail and soft highs`
- `massive poured concrete grain silo, rt60 around 27 seconds`
- `dark steel bunker, narrow early reflections`
- `reflective glass corridor with bright flutter`
- `intimate wooden chapel, warm and controlled`

## Limitations and Honesty

- v0 remains DSP-first with rule-based and lightweight learned conditioning.
- Acoustic metrics are engineering estimates for workflow consistency, not standards certification.
- Geometry-driven early reflections use image-source-lite first-order approximations, not full wave/geometric simulation.
- Spatial outputs are channel-layout projections; this is not an object-based Atmos renderer.

## Development Roadmap

See:

- [docs/architecture.md](docs/architecture.md)
- [docs/roadmap.md](docs/roadmap.md)
- [docs/benchmarking.md](docs/benchmarking.md)
- [docs/learned-encoders.md](docs/learned-encoders.md)
- [docs/perceptual-controls.md](docs/perceptual-controls.md)
- [docs/model-manifests.md](docs/model-manifests.md)
- [docs/spatial-layouts.md](docs/spatial-layouts.md)
- [docs/launch-readiness.md](docs/launch-readiness.md)

## Contributing

Contributions are welcome.

Recommended contributor workflow:

1. Open an issue describing proposed behavior and constraints.
2. Add/adjust tests for DSP behavior, schema outputs, and CLI UX.
3. Keep claims precise and technically defensible in docs.
4. Run `cargo fmt`, `cargo test`, and `./scripts/ci_regression_gates.sh` before submitting.

## Attribution

latent-ir is developed by Colby Leider and the open-source contributor community.

## License

MIT. See [LICENSE](LICENSE).
