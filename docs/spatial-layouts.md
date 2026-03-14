# Spatial Layouts

`latent-ir` supports built-in formats and arbitrary custom arrays.

## Built-in Formats

- `mono`
- `stereo`
- `foa` (ambiX order)
- `5.1`
- `7.1`
- `7.1.4`
- `7.2.4`

Built-ins emit stable channel labels and companion channel-map sidecars.

## Custom Layout Mode

Use:

```bash
latent-ir generate --channels custom --layout-json path/to/layout.json ...
```

Accepted per-channel fields:
- `label`
- `position_m` (`x`, `y`, `z`)
- `azimuth_deg`, `elevation_deg`
- optional `is_lfe`

Rules:
- provide either `position_m` or both polar angles
- if both cartesian and polar are provided, they must be consistent
- `position_m` is authoritative for geometry-driven processing when present

## Coordinate Convention

Cartesian -> polar derivation:
- `+Y = 0° azimuth`
- `+X = +90° azimuth`
- `+Z = elevation`

## Geometry-Driven Generation Behavior

With `position_m`, generation applies:
- relative propagation delay
- distance-based gain shaping
- distance-based HF air-loss shaping
- image-source-lite first-order early reflections

Optional virtual scene controls:
- `--source-x-m --source-y-m --source-z-m`
- `--listener-x-m --listener-y-m --listener-z-m`

## Example Layout Files

- `examples/layouts/custom_16_0_circle_10m_origin.json`
- `examples/layouts/custom_16_0_circle_10m_origin_cartesian_only.json`
- `examples/layouts/custom_abc_12.json`

## Example Command

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

## Current Non-Goals

- object-based Atmos scene rendering
- full geometric/wave room simulation
- HOA beyond current built-ins (use custom layouts for arbitrary channel counts)
