# Perceptual Controls

`latent-ir generate` now supports macro-level perceptual control primitives.

## Macros

All macro ranges are `[-1, 1]`.

- `--macro-size`: perceived room/space size
- `--macro-distance`: source-listener distance impression
- `--macro-material`: hardness/brightness of surfaces
- `--macro-clarity`: reflection clarity vs smearing/noise

Example:

```bash
cargo run -- generate \
  --prompt "cathedral" \
  --macro-size 0.8 \
  --macro-distance 0.4 \
  --macro-material -0.2 \
  --macro-clarity 0.3 \
  --output out/macro_ir.wav
```

## Trajectory automation

Use `--macro-trajectory` to automate macros over normalized time.

Trajectory schema:

```json
{
  "schema_version": "latent-ir.macro-trajectory.v1",
  "keyframes": [
    {"t": 0.0, "controls": {"size": -0.2, "distance": 0.0, "material": 0.0, "clarity": 0.2}},
    {"t": 1.0, "controls": {"size": 0.9, "distance": 0.5, "material": 0.3, "clarity": -0.1}}
  ]
}
```

Example:

```bash
cargo run -- generate \
  --prompt "impossible tunnel" \
  --macro-trajectory examples/macro_trajectory_rise.json \
  --output out/trajectory_ir.wav
```

The current implementation uses trajectory-conditioned segment synthesis and overlap blending for a dynamic IR evolution profile.
