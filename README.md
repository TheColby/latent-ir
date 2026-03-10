# latent-ir

**Generative impulse responses and acoustic spaces for the command line.**

`latent-ir` is a CLI-first Rust toolkit for generating, analyzing, morphing, and rendering impulse responses (IRs) using a hybrid architecture:

- ML-facing conditioning interfaces choose or infer acoustic intent.
- DSP builds, validates, and renders the resulting IR.

Current versions are procedural DSP + rule-based semantic conditioning. This repository is designed as a serious foundation for future learned modules, not a fake AI demo.

## Why This Exists

Most reverbs expose fixed presets or static measured spaces. `latent-ir` treats reverb as a controllable acoustic design space: scriptable, analyzable, reproducible, and extensible.

## Feature Overview (v0.1 scaffold)

- Descriptor-conditioned IR generation (`generate`)
- Engineering-focused IR analysis (`analyze`)
- IR morphing between two spaces (`morph`)
- Offline convolution rendering (`render`)
- Descriptor sampling utilities (`sample`)
- Built-in preset inspection and retrieval (`preset`)
- Deterministic generation via explicit seed
- JSON metadata + analysis output for reproducibility

## Architecture Overview

High-level pipeline:

```
prompt / reference / explicit params
-> conditioning layer (v0: rule-based semantics + presets)
-> DescriptorSet (canonical acoustic state)
-> ProceduralIrGenerator (DSP)
-> normalization + sanity checks
-> final IR
-> IrAnalyzer + JSON metadata
```

Core modules:

- `src/core/descriptors`: canonical acoustic representation
- `src/core/semantics`: prompt-token -> descriptor adjustments
- `src/core/presets`: named descriptor defaults
- `src/core/generator`: procedural IR synthesis engine
- `src/core/analysis`: first-pass acoustics metrics
- `src/core/morph`: IR and descriptor interpolation
- `src/core/render`: offline convolution rendering

## Current Status

This is an initial but functional scaffold with real behavior in core commands.

Implemented now:

- procedural mono/stereo IR generation with direct impulse, early reflections, dense stochastic tail, simple band-dependent decay shaping, deterministic seed
- approximate T20/T30/T60/EDT/predelay/EDC-centric analysis
- spectral centroid and low/mid/high decay estimates
- stereo correlation + early/late energy summaries
- offline WAV convolution with wet/dry control

Not implemented yet:

- learned text/audio encoders
- latent diffusion/autoencoder style IR synthesis
- standards-certified architectural acoustics reporting
- geometric or wave-based room simulation

## Installation

Prerequisites:

- Rust stable toolchain (`rustup`, `cargo`)

Build:

```bash
cargo build --release
```

Run help:

```bash
cargo run -- --help
```

## Build and Test

```bash
cargo fmt
cargo test
```

## Usage Examples

Generate cathedral-like IR:

```bash
cargo run -- generate \
  --prompt "vast icy cathedral" \
  --t60 12 \
  --channels stereo \
  --seed 1337 \
  --output out/cathedral_ir.wav
```

Generate from preset + overrides:

```bash
cargo run -- generate \
  --preset steel_bunker \
  --duration 5.5 \
  --predelay-ms 18 \
  --output out/bunker_ir.wav
```

Analyze IR with JSON output:

```bash
cargo run -- analyze out/cathedral_ir.wav --json
```

Morph two IRs:

```bash
cargo run -- morph cave_ir.wav plate_ir.wav --alpha 0.4 --output out/morphed.wav
```

Render dry signal through IR:

```bash
cargo run -- render dry_input.wav --ir out/morphed.wav --mix 0.25 --output out/rendered.wav
```

List presets:

```bash
cargo run -- preset
```

Inspect one preset:

```bash
cargo run -- preset dark_stone_cathedral --json
```

## Command Reference

- `generate`
  - Inputs: optional `--prompt`, optional `--preset`, descriptor overrides (`--duration`, `--t60`, `--predelay-ms`, `--edt`, etc.), channel format, seed
  - Outputs: generated IR WAV + companion JSON metadata (or custom `--metadata-out`), optional analysis JSON via `--json-analysis-out`
- `analyze`
  - Inputs: IR WAV
  - Outputs: console metrics or JSON analysis report
- `morph`
  - Inputs: two IR WAVs + `--alpha`
  - Outputs: morphed IR WAV
- `render`
  - Inputs: dry WAV + `--ir` + `--mix`
  - Outputs: rendered WAV
- `sample`
  - Inputs: count + seed
  - Outputs: random descriptor samples (text or JSON)
- `preset`
  - Inputs: optional preset name
  - Outputs: preset list or descriptor JSON

## Limitations and Honesty

- v0 uses procedural DSP and rule-based prompt semantics only.
- Prompt interpretation is deterministic token matching, not an LLM/transformer encoder.
- Acoustic metrics are engineering estimates for development and comparison, not certified room-acoustics measurements.
- IR generation is synthetic and intentionally practical; it does not claim full physical room simulation.

## JSON Schema Stability (v0)

- Generation metadata includes `schema_version: "latent-ir.generation.v1"`.
- Analysis reports include `schema_version: "latent-ir.analysis.v1"`.
- These version tags are intended for machine parsing and forward compatibility as report fields evolve.

## Hybrid DSP + ML Direction

`latent-ir` is explicitly designed for future learned modules:

- learned text/audio conditioning can plug into descriptor inference
- latent model outputs can be constrained/projected through descriptor validators
- DSP synthesis and analysis remain first-class and auditable

This keeps results controllable and reproducible while still opening a path to modern generative models.

## Roadmap

See:

- `docs/architecture.md`
- `docs/roadmap.md`

## Contributing

Contributions are welcome, especially in:

- acoustics metric quality
- DSP model quality/performance
- reproducibility and metadata standards
- future ML interface design

Please open an issue describing:

- expected behavior
- input assets and sample rates
- command invocation and output
- whether behavior is regression vs expected limitation

## Example Prompts and Presets

Prompt ideas:

- `"intimate wood chapel"`
- `"dark stone cathedral"`
- `"steel bunker"`
- `"bright glass corridor"`
- `"impossible infinite tunnel"`

Built-in presets:

- `intimate_wood_chapel`
- `dark_stone_cathedral`
- `steel_bunker`
- `glass_corridor`
- `frozen_plate`
- `impossible_infinite_tunnel`
