# Spatial Layouts

`latent-ir` supports built-in surround/ambisonic-style layouts and custom user-defined arrays.

## Built-in Formats

- `mono`
- `stereo`
- `foa` (ambiX channel order)
- `5.1`
- `7.1`
- `7.1.4`
- `7.2.4`

Built-ins include stable labels and a generated channel-map sidecar for reproducible downstream analysis.

## Custom Layouts

Use:

```bash
latent-ir generate --channels custom --layout-json path/to/layout.json ...
```

The layout schema supports:

- `position_m`: cartesian coordinates in meters (`x`, `y`, `z`)
- `azimuth_deg` and `elevation_deg`: polar metadata
- optional `is_lfe` flag per channel

When `position_m` is present, it is treated as authoritative for geometry-driven processing.

## Coordinate Convention

Cartesian to polar derivation follows:

- `+Y = 0 deg azimuth`
- `+X = +90 deg azimuth`
- `+Z = elevation`

This convention is validated when both cartesian and polar values are provided.

## Geometry-Driven Behavior

For custom layouts with `position_m`, generation currently applies:

- relative propagation delay by distance
- distance-dependent gain falloff
- distance-dependent HF air-loss shaping
- image-source-lite early reflection clusters

Virtual source and listener positions can be supplied via:

- `--source-x-m --source-y-m --source-z-m`
- `--listener-x-m --listener-y-m --listener-z-m`

## Example: 16-Channel Ring

See:

- `examples/layouts/custom_16_0_circle_10m_origin.json`
- `examples/layouts/custom_16_0_circle_10m_origin_cartesian_only.json`

Example command:

```bash
cargo run -- generate \
  --prompt "32-second dark cave" \
  --duration 32 \
  --sample-rate 384000 \
  --channels custom \
  --layout-json examples/layouts/custom_16_0_circle_10m_origin_cartesian_only.json \
  --source-x-m 0.0 --source-y-m 24.0 --source-z-m 2.0 \
  --listener-x-m 0.0 --listener-y-m 0.0 --listener-z-m 1.5 \
  --output out/dark_cave_16ch_cart_384k.wav
```

## Out of Scope (Current)

- Object-based Atmos scene rendering
- HOA orders beyond currently hardcoded built-ins (use `custom` for arbitrary arrays)
- Full geometric or wave-based room simulation
