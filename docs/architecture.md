# latent-ir Architecture

## Design Thesis

`latent-ir` models reverberation as a descriptor-conditioned acoustic design space.

Guiding principle:

**Conditioning proposes intent; DSP synthesis enforces behavior.**

This keeps generation deterministic and inspectable while still leaving room for learned models.

## End-to-End Pipeline

```text
prompt / reference audio / explicit params
-> conditioning chain
-> DescriptorSet
-> procedural generator
-> validation / normalization
-> final IR
-> analyzer + metadata sidecars
```

## Canonical Representation (`DescriptorSet`)

The descriptor model is the compatibility contract across CLI, synthesis, analysis, and learning.

Domains:
- time: `duration`, `predelay`, `t60`, `edt`, `attack_gap`
- spectral: brightness/damping/tilt + low/mid/high decay controls
- structural: early/late density, diffusion, modal/tail/grain controls
- spatial: channel format, width/decorrelation/asymmetry, custom layout, optional source/listener positions

## Conditioning Stack

Current stack can combine:
- preset defaults
- semantic prompt resolver (rule-based)
- learned text encoder (JSON model, optional ONNX adapter)
- learned audio encoder (JSON model, optional ONNX adapter)
- explicit CLI overrides

All deltas are resolved onto `DescriptorSet` before synthesis.

## Synthesis Model (v0.x)

Procedural IR decomposition:
1. direct impulse anchor + predelay
2. sparse early reflections
3. dense stochastic late tail
4. frequency-dependent envelope shaping
5. channel projection for built-in and custom layouts
6. geometry-aware custom processing (`position_m`):
   - relative delay
   - distance gain
   - HF air-loss shaping
   - image-source-lite first-order early clusters
7. normalization and sanity checks

Tail behavior guardrail:
- generation applies a duration floor by default to reduce premature tail truncation
- user can opt out via `--allow-tail-truncation`

## Spatial Model

Built-in layouts:
- `mono`, `stereo`, `foa` (ambiX), `5.1`, `7.1`, `7.1.4`, `7.2.4`

Custom layout JSON supports:
- cartesian (`position_m`)
- polar (`azimuth_deg`, `elevation_deg`)
- mixed input with consistency validation

Cartesian convention:
- `+Y = 0° azimuth`
- `+X = +90° azimuth`
- `+Z = elevation`

`generate` emits a validated channel map sidecar (`*.channels.json`) for reproducible downstream routing and analysis.

## Analysis Philosophy

Analysis is deterministic and workflow-oriented.

Key outputs include:
- EDT / T20 / T30 / T60 estimates
- decay-span and confidence estimates for decay metrics
- predelay estimate
- spectral centroid + band decay summaries
- early/late energy summaries
- inter-channel correlation summaries
- arrival spread, ITD-ish, IACC-style coherence metrics
- directional energy summaries when channel map is available

These values are engineering estimates for workflow consistency, not standards-certified architectural acoustics metrology.

## Render Architecture

Render engines:
- `direct`
- `fft-partitioned`
- `fft-streaming`

`render --engine auto` chooses engine by workload heuristics and reports the decision.

Sample-rate mismatch handling:
- strict by default (explicit error)
- optional auto reconciliation in `render` and `morph` via `--auto-resample`

## Reproducibility Contract

A run should be reproducible from:
- command args
- seed
- sample rate
- resolved descriptor values
- model paths/manifests
- project version

Artifacts:
- IR WAV
- generation metadata JSON (`latent-ir.generation.v1`)
- analysis JSON (`latent-ir.analysis.v1`)
- channel map JSON (`latent-ir.channel-map.v1`)

Generation metadata includes:
- replay command string
- conditioning trace (text/audio/combined deltas)
- warnings and embedded analysis summary

`generate --explain-conditioning` provides an interactive console view of this conditioning state.

## Extension Points

Planned near-term expansion:
- latent projection contracts
- confidence/uncertainty propagation
- descriptor trajectory coupling for morph/generation
- broader spatial benchmark coverage
