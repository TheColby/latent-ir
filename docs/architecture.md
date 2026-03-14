# latent-ir Architecture

## Thesis

`latent-ir` models reverberation as a descriptor-conditioned acoustic design space.

Core principle:

**ML selects or infers intent; DSP synthesizes and validates the IR.**

The system stays deterministic and auditable even when learned modules are present.

## System Pipeline

```text
prompt / reference audio / explicit params
-> conditioning chain
-> DescriptorSet
-> ProceduralIrGenerator
-> normalization + constraints
-> final IR
-> IrAnalyzer + JSON sidecars
```

## Canonical Acoustic Representation

`DescriptorSet` groups parameters into four domains:

- Time: `duration`, `predelay`, `t60`, `edt`, `attack_gap`
- Spectral: `brightness`, `hf_damping`, `lf_bloom`, `spectral_tilt`, band decays
- Structural: early/late density, diffusion, modal density, tail noise, grain
- Spatial: format, width, decorrelation, asymmetry, custom layout geometry, optional source/listener positions

This representation is the interoperability contract across generation, analysis, learning, and metadata.

## Conditioning Layers

Current chain can combine:

- preset defaults
- rule-based semantic resolver
- learned text encoder (JSON model; optional ONNX backend)
- learned audio encoder (JSON model; optional ONNX backend)
- explicit CLI overrides

Conditioning outputs descriptor deltas that are applied before clamping and generation.

## Procedural IR Synthesis

Main components:

1. direct impulse/predelay anchor
2. sparse early reflections
3. dense stochastic late tail
4. frequency-dependent envelope shaping
5. spatial projection to built-in or custom channel layouts
6. geometry-aware custom shaping from `position_m`:
   - relative propagation delay
   - distance gain falloff
   - HF air-loss
   - image-source-lite first-order early reflections
7. optional virtual source/listener steering via `source_position_m` / `listener_position_m`
8. normalization and output sanity checks

## Spatial Model

Built-in layouts:

- `mono`, `stereo`, `foa` (ambiX), `5.1`, `7.1`, `7.1.4`, `7.2.4`

Custom layout JSON supports:

- polar (`azimuth_deg` + `elevation_deg`)
- cartesian (`position_m`)
- mixed inputs with consistency validation

Coordinate convention for cartesian->polar derivation:

- `+Y = 0 deg azimuth`
- `+X = +90 deg azimuth`
- `+Z = elevation`

`generate` emits validated `*.channels.json` channel-map sidecars for downstream analysis/routing reproducibility.

## Analysis Philosophy

Metrics are intentionally engineering-focused and deterministic:

- EDC decay estimates: EDT/T20/T30/T60
- predelay estimate
- spectral centroid + band decay summaries
- early/late energy ratios
- stereo and inter-channel correlation summaries
- directional energy summaries (front/rear/height/LFE with channel map)
- arrival spread summaries
- ITD-ish and IACC-style early coherence metrics

These are not standards-certified architectural acoustics measurements.

Both `generate` and `analyze` surface this caveat directly in console output so downstream users do not mistake these values for certified metrology.

## Render Strategy

Render supports three concrete engines:

- `direct`
- `fft-partitioned`
- `fft-streaming`

`render --engine auto` selects an engine from input/IR/channel workload heuristics and reports the decision in console output. This keeps small jobs simple while scaling large multichannel jobs with lower memory pressure.

## Reproducibility Model

A run is intended to be reproducible from:

- command arguments
- seed
- sample rate
- resolved descriptor state
- model files/manifests used
- project version

Artifacts:

- WAV output
- generation metadata JSON (`latent-ir.generation.v1`)
- analysis report (`latent-ir.analysis.v1`)
- channel map sidecar (`latent-ir.channel-map.v1`)

## Forward Compatibility

Planned extensibility points:

- richer latent conditioning contracts
- confidence/uncertainty propagation
- trajectory-conditioned morph/generation
- stronger spatial/object/array abstractions
