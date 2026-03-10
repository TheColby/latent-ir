<p align="center">
  <img src="docs/assets/latent-ir-logo.svg" alt="latent-ir logo" width="760" />
</p>

# latent-ir

**Generative impulse responses and acoustic spaces for the command line.**

`latent-ir` is a CLI-first Rust toolkit for generating, analyzing, morphing, and rendering impulse responses (IRs) using a hybrid architecture:

- ML-facing conditioning interfaces choose or infer acoustic intent.
- DSP builds, validates, and renders the resulting IR.

Current versions are procedural DSP + rule-based semantic conditioning. This repository is designed as a serious foundation for future learned modules, not a fake AI demo.

## Why This Exists

Most reverbs expose fixed presets or static measured spaces. `latent-ir` treats reverb as a controllable acoustic design space: scriptable, analyzable, reproducible, and extensible.

## Feature Overview (current scaffold)

- Descriptor-conditioned IR generation (`generate`)
- Spatial layout generation for `mono`, `stereo`, `foa` (ambiX), `5.1`, `7.1`, `7.1.4`, `7.2.4`, and `custom` layout JSON
- Validated channel-map sidecars (`.channels.json`) for reproducible spatial routing
- Engineering-focused IR analysis (`analyze`)
- Multichannel analysis metrics (inter-channel correlation matrix + directional/LFE energy balance)
- IR morphing between two spaces (`morph`)
- Offline convolution rendering (`render`)
- Descriptor sampling utilities (`sample`)
- Built-in preset inspection and retrieval (`preset`)
- Model fitting utility for lightweight learned encoders (`train-encoder`)
- Evaluation utility with baseline reports for regression tracking (`eval`)
- Benchmark lab + CI gating utility (`benchmark`)
- Model manifest validation utility (`model validate`)
- One-shot industrial-vs-baseline comparison utility (`ab-test`)
- Learned text/audio conditioning encoders loaded from JSON (plus optional ONNX backend)
- Deterministic generation via explicit seed
- JSON metadata + analysis output for reproducibility, including channel format, spatial encoding, and channel labels

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
- `src/core/spatial`: custom layout parsing, validation, and channel-map sidecars
- `src/core/morph`: IR and descriptor interpolation
- `src/core/render`: offline convolution rendering

## Current Status

This is an initial but functional scaffold with real behavior in core commands.

Implemented now:

- procedural IR generation for mono/stereo + multichannel layouts (`foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`, `custom`) with direct impulse, early reflections, dense stochastic tail, simple band-dependent decay shaping, deterministic seed
- validated spatial channel-map JSON sidecars written on generation and auto-consumed by analysis when present
- approximate T20/T30/T60/EDT/predelay/EDC-centric analysis
- spectral centroid and low/mid/high decay estimates
- stereo pair correlation (channel 0/1), inter-channel correlation matrix, and directional energy summaries (front/rear/height/LFE with channel map)
- offline WAV convolution with wet/dry control and optimized FFT partitioned engine for high channel counts

Not implemented yet:

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

Generate FOA ambisonic IR (ambiX channel order):

```bash
cargo run -- generate \
  --prompt "massive concrete grain silo" \
  --channels foa \
  --output out/silo_foa.wav
```

Generate Atmos-style 7.2.4 bed IR:

```bash
cargo run -- generate \
  --prompt "industrial aircraft hangar" \
  --channels 7.2.4 \
  --output out/hangar_724.wav
```

Generate 5.1 and 7.1.4 variants from the same prompt:

```bash
cargo run -- generate --prompt "dark concrete transit terminal" --channels 5.1 --output out/terminal_51.wav
cargo run -- generate --prompt "dark concrete transit terminal" --channels 7.1.4 --output out/terminal_714.wav
```

Generate custom 16.0 from layout JSON:

```bash
cargo run -- generate \
  --prompt "circular concrete tank with long smooth tail" \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_ring.json \
  --output out/tank_16_0.wav
```

Generate a 16-channel IR on a 10 m circular array (capture at origin), using prompt `"32-second dark cave"` at `384000` Hz:

```bash
cargo run -- generate \
  --prompt "32-second dark cave" \
  --duration 32 \
  --sample-rate 384000 \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_circle_10m_origin.json \
  --output out/dark_cave_16ch_384k.wav
```

Corresponding array-geometry JSON file:

```json
{
  "schema_version": "latent-ir.layout.v1",
  "layout_name": "custom_16_0_circle_10m_origin",
  "spatial_encoding": "discrete",
  "array_geometry": {
    "shape": "circle",
    "radius_m": 10.0,
    "center_m": { "x": 0.0, "y": 0.0, "z": 0.0 },
    "capture_position_m": { "x": 0.0, "y": 0.0, "z": 0.0 },
    "notes": "16 channels equally distributed around a 10 m radius circle; IR capture at origin"
  },
  "channels": [
    { "label": "C00", "azimuth_deg": 0, "elevation_deg": 0, "position_m": { "x": 10.0000, "y": 0.0000, "z": 0.0 } },
    { "label": "C01", "azimuth_deg": 23, "elevation_deg": 0, "position_m": { "x": 9.2388, "y": 3.8268, "z": 0.0 } },
    { "label": "C02", "azimuth_deg": 45, "elevation_deg": 0, "position_m": { "x": 7.0711, "y": 7.0711, "z": 0.0 } },
    { "label": "C03", "azimuth_deg": 68, "elevation_deg": 0, "position_m": { "x": 3.8268, "y": 9.2388, "z": 0.0 } },
    { "label": "C04", "azimuth_deg": 90, "elevation_deg": 0, "position_m": { "x": 0.0000, "y": 10.0000, "z": 0.0 } },
    { "label": "C05", "azimuth_deg": 113, "elevation_deg": 0, "position_m": { "x": -3.8268, "y": 9.2388, "z": 0.0 } },
    { "label": "C06", "azimuth_deg": 135, "elevation_deg": 0, "position_m": { "x": -7.0711, "y": 7.0711, "z": 0.0 } },
    { "label": "C07", "azimuth_deg": 158, "elevation_deg": 0, "position_m": { "x": -9.2388, "y": 3.8268, "z": 0.0 } },
    { "label": "C08", "azimuth_deg": 180, "elevation_deg": 0, "position_m": { "x": -10.0000, "y": 0.0000, "z": 0.0 } },
    { "label": "C09", "azimuth_deg": -158, "elevation_deg": 0, "position_m": { "x": -9.2388, "y": -3.8268, "z": 0.0 } },
    { "label": "C10", "azimuth_deg": -135, "elevation_deg": 0, "position_m": { "x": -7.0711, "y": -7.0711, "z": 0.0 } },
    { "label": "C11", "azimuth_deg": -113, "elevation_deg": 0, "position_m": { "x": -3.8268, "y": -9.2388, "z": 0.0 } },
    { "label": "C12", "azimuth_deg": -90, "elevation_deg": 0, "position_m": { "x": 0.0000, "y": -10.0000, "z": 0.0 } },
    { "label": "C13", "azimuth_deg": -68, "elevation_deg": 0, "position_m": { "x": 3.8268, "y": -9.2388, "z": 0.0 } },
    { "label": "C14", "azimuth_deg": -45, "elevation_deg": 0, "position_m": { "x": 7.0711, "y": -7.0711, "z": 0.0 } },
    { "label": "C15", "azimuth_deg": -23, "elevation_deg": 0, "position_m": { "x": 9.2388, "y": -3.8268, "z": 0.0 } }
  ]
}
```

The generator currently uses `azimuth_deg`/`elevation_deg` (polar) for synthesis. `position_m` is included for explicit geometry documentation/interchange.

Generate A.B.C-style custom layout from layout JSON:

```bash
cargo run -- generate \
  --prompt "multi-zone reflective chamber" \
  --channels custom \
  --layout-json examples/layouts/custom_abc_12.json \
  --output out/chamber_abc.wav
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

## Spatial Audio Examples

FOA (ambiX `WXYZ`) generation + analysis:

```bash
cargo run -- generate \
  --prompt "massive cylindrical grain silo, thick concrete, very long decay" \
  --channels foa \
  --metadata-out out/silo_foa.meta.json \
  --output out/silo_foa.wav

cargo run -- analyze out/silo_foa.wav --json --output out/silo_foa.analysis.json
```

Atmos-style 7.2.4 bed generation + render:

```bash
cargo run -- generate \
  --prompt "aircraft hangar with long metallic late tail" \
  --channels 7.2.4 \
  --output out/hangar_724_ir.wav \
  --metadata-out out/hangar_724_ir.json

cargo run -- render \
  dry_724.wav \
  --ir out/hangar_724_ir.wav \
  --engine fft-partitioned \
  --partition-size 2048 \
  --mix 0.3 \
  --output out/hangar_724_render.wav
```

Spatial metadata fields (`channel_format`, `spatial_encoding`, `channel_labels`) are emitted in generation JSON for downstream DAW/tooling workflows.
Each generated IR also writes a validated companion channel-map sidecar (`*.channels.json`) unless `--channel-map-out` is set.

## Command Reference

- `generate`
  - Inputs: optional `--prompt`, optional `--preset`, optional learned encoders (`--text-encoder-model`, `--audio-encoder-model`, `--text-encoder-onnx`, `--audio-encoder-onnx`, `--reference-audio`), descriptor overrides (`--duration`, `--t60`, `--predelay-ms`, `--edt`, etc.), channel format (`mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`, `custom`), optional `--layout-json` for custom layouts, seed
  - Perceptual controls: `--macro-size`, `--macro-distance`, `--macro-material`, `--macro-clarity`, `--macro-trajectory`
  - Outputs: generated IR WAV + companion JSON metadata (or custom `--metadata-out`), validated channel-map JSON sidecar (or custom `--channel-map-out`), optional analysis JSON via `--json-analysis-out`
  - Console: prints detailed IR/reverb metrics for every generation run (layout, labels, length, EDT/T20/T30/T60, predelay, spectral, energy split, stereo pair correlation)
- `analyze`
  - Inputs: IR WAV, optional explicit channel map via `--channel-map` (otherwise companion sidecar is auto-detected)
  - Outputs: console metrics or JSON analysis report
- `morph`
  - Inputs: two IR WAVs + `--alpha`
  - Outputs: morphed IR WAV
- `render`
  - Inputs: dry WAV + `--ir` + `--mix`
  - Optional performance controls: `--engine auto|direct|fft-partitioned`, `--partition-size` (FFT engine is optimized for multichannel/high-order layouts)
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
- `ab-test`
  - Runs paired generation + analysis for industrial-model vs baseline and writes `latent-ir.ab-test.v1` report with deltas
  - Optional `--markdown` writes `ab_test_report.md` scorecard

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

Run one-shot industrial-vs-baseline A/B test:

```bash
cargo run -- ab-test \
  --prompt "massive, colossal grain silo made of 3 ft poured concrete, with an RT60 of around 27 seconds" \
  --industrial-text-model models/text_encoder_industrial_v1.json \
  --t60 27 \
  --macro-size 1.0 \
  --macro-distance 0.8 \
  --macro-material -0.2 \
  --markdown \
  --output-dir out/ab_silo
```

## Limitations and Honesty

- v0 uses procedural DSP and rule-based prompt semantics only.
- Prompt interpretation is deterministic token matching, not an LLM/transformer encoder.
- Acoustic metrics are engineering estimates for development and comparison, not certified room-acoustics measurements.
- IR generation is synthetic and intentionally practical; it does not claim full physical room simulation.
- Spatial outputs are layout-projected synthetic IRs; current support is not an object-based Atmos renderer or standards-certified ambisonic room simulation.
- Custom layouts are only as meaningful as their provided channel geometry; always define explicit azimuth/elevation/LFE semantics.

## JSON Schema Stability (v0)

- Generation metadata includes `schema_version: "latent-ir.generation.v1"`.
- Analysis reports include `schema_version: "latent-ir.analysis.v1"`.
- Evaluation baseline reports include `schema_version: "latent-ir.eval.baseline.v1"`.
- Benchmark reports include `schema_version: "latent-ir.benchmark.v1"`.
- Model manifests use `schema_version: "latent-ir.model-manifest.v1"`.
- A/B reports include `schema_version: "latent-ir.ab-test.v1"`.
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

- `"massive, colossal grain silo made of 3 ft poured concrete, rt60 27 seconds"`
- `"narrow maintenance tunnel with corrugated steel and harsh flutter"`
- `"abandoned underground cistern, booming low end, very long tail"`
- `"small machine room with concrete walls and moderate metallic ring"`
- `"distant source in a huge empty aircraft hangar, dark and diffuse"`

Built-in presets:

- `intimate_wood_chapel`
- `dark_stone_cathedral`
- `steel_bunker`
- `glass_corridor`
- `frozen_plate`
- `impossible_infinite_tunnel`

## Attribution

`latent-ir` is developed by Colby Leider and the open-source contributor community.

If you use this project in research, products, demos, or educational material, attribution to the `latent-ir` project and repository is appreciated.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for the full license text.
