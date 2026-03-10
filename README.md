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
- Model fitting utility for lightweight learned encoders (`train-encoder`)
- Evaluation utility with baseline reports for regression tracking (`eval`)
- Benchmark lab + CI gating utility (`benchmark`)
- Model manifest validation utility (`model validate`)
- Learned text/audio conditioning encoders loaded from JSON model files
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

Generate with learned text encoder:

```bash
cargo run -- generate \
  --prompt "dark steel cathedral" \
  --text-encoder-model examples/models/text_encoder_v1.json \
  --output out/learned_text_ir.wav
```

Generate with ONNX text encoder (build with `--features onnx`):

```bash
cargo run --features onnx -- generate \
  --prompt "dark steel cathedral" \
  --text-encoder-onnx models/text_delta.onnx \
  --text-encoder-onnx-input-dim 256 \
  --output out/onnx_text_ir.wav
```

Generate with learned audio encoder from reference material:

```bash
cargo run -- generate \
  --reference-audio references/hall_ir.wav \
  --audio-encoder-model examples/models/audio_encoder_v1.json \
  --output out/learned_audio_ir.wav
```

Generate with ONNX audio encoder (build with `--features onnx`):

```bash
cargo run --features onnx -- generate \
  --reference-audio references/hall_ir.wav \
  --audio-encoder-onnx models/audio_delta.onnx \
  --output out/onnx_audio_ir.wav
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

Render using FFT partitioned convolution:

```bash
cargo run -- render dry_input.wav \
  --ir out/morphed.wav \
  --engine fft-partitioned \
  --partition-size 2048 \
  --mix 0.25 \
  --output out/rendered_fft.wav
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
  - Inputs: optional `--prompt`, optional `--preset`, optional learned encoders (`--text-encoder-model`, `--audio-encoder-model`, `--text-encoder-onnx`, `--audio-encoder-onnx`, `--reference-audio`), descriptor overrides (`--duration`, `--t60`, `--predelay-ms`, `--edt`, etc.), channel format, seed
  - Perceptual controls: `--macro-size`, `--macro-distance`, `--macro-material`, `--macro-clarity`, `--macro-trajectory`
  - Outputs: generated IR WAV + companion JSON metadata (or custom `--metadata-out`), optional analysis JSON via `--json-analysis-out`
- `analyze`
  - Inputs: IR WAV
  - Outputs: console metrics or JSON analysis report
- `morph`
  - Inputs: two IR WAVs + `--alpha`
  - Outputs: morphed IR WAV
- `render`
  - Inputs: dry WAV + `--ir` + `--mix`
  - Optional performance controls: `--engine auto|direct|fft-partitioned`, `--partition-size`
  - Outputs: rendered WAV
- `sample`
  - Inputs: count + seed
  - Outputs: random descriptor samples (text or JSON)
- `preset`
  - Inputs: optional preset name
  - Outputs: preset list or descriptor JSON
- `train-encoder`
  - Modes: `text`, `audio`
  - Outputs: learned model JSON compatible with `--text-encoder-model` / `--audio-encoder-model`
- `eval`
  - Modes: `text`, `audio`
  - Outputs: baseline evaluation JSON (`latent-ir.eval.baseline.v1`) with descriptor-space and analysis-space errors
- `benchmark`
  - `run`: computes objective/speed/stability/perceptual-proxy metrics and writes `latent-ir.benchmark.v1` report
  - `check`: compares report vs baseline and fails on configured regression thresholds
- `model`
  - `validate`: validates `latent-ir.model-manifest.v1` manifests against runtime capabilities/features

Train text encoder from labeled prompt data:

```bash
cargo run -- train-encoder text \
  --dataset datasets/text_pairs.json \
  --output models/text_encoder_v1.json \
  --max-vocab 512 \
  --epochs 800
```

Train audio encoder from labeled reference-audio data:

```bash
cargo run -- train-encoder audio \
  --dataset datasets/audio_pairs.json \
  --output models/audio_encoder_v1.json \
  --epochs 1000
```

Evaluate text encoder on held-out data:

```bash
cargo run -- eval text \
  --dataset datasets/text_eval.json \
  --model models/text_encoder_v1.json \
  --output reports/text_baseline.json
```

Evaluate audio encoder on held-out data:

```bash
cargo run -- eval audio \
  --dataset datasets/audio_eval.json \
  --model models/audio_encoder_v1.json \
  --output reports/audio_baseline.json
```

Run benchmark suite:

```bash
cargo run -- benchmark run \
  --dataset datasets/benchmark_suite.json \
  --text-model models/text_encoder_v1.json \
  --audio-model models/audio_encoder_v1.json \
  --repeats 3 \
  --output reports/benchmark.json
```

Gate candidate report vs baseline:

```bash
cargo run -- benchmark check \
  --report reports/benchmark_new.json \
  --baseline reports/benchmark_baseline.json \
  --max-regression 0.05
```

Validate model/runtime compatibility:

```bash
cargo run -- model validate --manifest manifests/text_encoder_manifest.json
```

## Limitations and Honesty

- v0 uses procedural DSP and rule-based prompt semantics only.
- Prompt interpretation is deterministic token matching, not an LLM/transformer encoder.
- Acoustic metrics are engineering estimates for development and comparison, not certified room-acoustics measurements.
- IR generation is synthetic and intentionally practical; it does not claim full physical room simulation.

## JSON Schema Stability (v0)

- Generation metadata includes `schema_version: "latent-ir.generation.v1"`.
- Analysis reports include `schema_version: "latent-ir.analysis.v1"`.
- Evaluation baseline reports include `schema_version: "latent-ir.eval.baseline.v1"`.
- Benchmark reports include `schema_version: "latent-ir.benchmark.v1"`.
- Model manifests use `schema_version: "latent-ir.model-manifest.v1"`.
- These version tags are intended for machine parsing and forward compatibility as report fields evolve.

## Hybrid DSP + ML Direction

`latent-ir` is explicitly designed for future learned modules:

- learned text/audio conditioning can plug into descriptor inference
- latent model outputs can be constrained/projected through descriptor validators
- DSP synthesis and analysis remain first-class and auditable

This keeps results controllable and reproducible while still opening a path to modern generative models.

## Learned Encoder Models (v0.1)

`latent-ir` now includes first-pass learned conditioning hooks:

- `LearnedTextEncoder`: token embedding table + learned linear projection to descriptor deltas
- `LearnedAudioEncoder`: engineered audio feature frontend + learned MLP + projection to descriptor deltas

Both are loaded from JSON model files and run entirely in Rust. Example model files are provided in `examples/models/`.
Training CLI support is provided via `train-encoder`.

## Roadmap

See:

- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/learned-encoders.md`
- `docs/benchmarking.md`
- `docs/perceptual-controls.md`
- `docs/model-manifests.md`

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

## Attribution

`latent-ir` is developed by the open-source contributor community.

If you use this project in research, products, demos, or educational material, attribution to the `latent-ir` project and repository is appreciated.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for the full license text.
