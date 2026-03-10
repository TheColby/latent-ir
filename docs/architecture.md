# latent-ir Architecture

## Thesis

`latent-ir` treats reverberation as a descriptor-conditioned design space rather than a fixed IR library.

Core principle:

**ML chooses acoustic behavior, DSP builds and validates the IR.**

In v0, the ML side is represented by explicit extension interfaces and rule-based semantic conditioning. DSP is fully active and generates real outputs.
v0.1 also includes learned text/audio encoder hooks that load small model weights from JSON and infer descriptor deltas directly in the CLI pipeline.

## Descriptor-Conditioned Generation

Canonical acoustic state is represented by `DescriptorSet` with grouped domains:

- Time: duration, predelay, T60, EDT, attack gap
- Spectral: brightness, HF damping, LF bloom, spectral tilt, low/mid/high decay multipliers
- Structural: early/late density, diffusion, modal density, tail noise, grain
- Spatial: channel format, width, decorrelation, asymmetry

Generation pipeline resolves descriptor values from:

1. built-in preset (optional)
2. semantic prompt token mapping (optional)
3. explicit CLI overrides
4. range clamps and normalization

## Procedural IR Decomposition

The procedural generator synthesizes IRs as layered components:

1. direct impulse at predelay
2. sparse early reflection field with density and stereo spread
3. dense late stochastic tail
4. frequency-weighted envelope shaping via band-dependent decay constants
5. stereo decorrelation jitter and width shaping
6. output normalization and sanity bounds

This is intentionally practical, deterministic, and auditable.

## Semantic Prompt Conditioning (v0)

`SemanticResolver` provides deterministic token-based rules (e.g. `cathedral`, `steel`, `wood`, `dark`, `bright`, `infinite`).

This is explicitly not a learned text model. It exists to validate CLI flow and descriptor-conditioning architecture while keeping outputs reproducible.

In parallel, `LearnedTextEncoder` can be supplied in `generate` to apply learned prompt-conditioned descriptor deltas before semantic rules and explicit overrides.

## Future ML Module Interfaces

Planned extension path:

- `ConditioningModel` traits for text/audio/reference inference
- richer learned text/audio models (ONNX, safetensors, or custom backends)
- learned encoders producing descriptor priors or latent embeddings
- latent -> descriptor projection with constraint checks
- hybrid generator modes combining latent proposals with DSP enforcement

Guiding constraint: learned components should be optional and never remove deterministic DSP fallback.

## Analysis Philosophy

v0 analysis targets robust engineering estimates suitable for CLI workflows:

- duration / peak / RMS
- EDC-based decay estimates (EDT, T20, T30, T60)
- predelay estimate
- spectral centroid summary
- coarse low/mid/high decay splits
- stereo correlation
- early-vs-late energy ratio

Metrics are intentionally labeled as approximations where standards-grade compliance is not yet implemented.

## Reproducibility Philosophy

Every generation run should be reproducible from:

- seed
- resolved descriptor set
- sample rate
- command context
- project version

`generate` writes companion JSON metadata with analysis report and warnings. This makes batch runs scriptable and traceable for both research and product workflows.
