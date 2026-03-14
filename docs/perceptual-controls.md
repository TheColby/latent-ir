# Perceptual Controls

`latent-ir generate` exposes macro-level controls for fast descriptor steering.

## Macro Controls

Range for each macro is `[-1, 1]`.

- `--macro-size`: perceived space size
- `--macro-distance`: perceived source-listener distance
- `--macro-material`: perceived material hardness/brightness
- `--macro-clarity`: perceived clarity vs smearing

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

## Trajectory Automation

`--macro-trajectory` accepts a normalized-time keyframe file:

```json
{
  "schema_version": "latent-ir.macro-trajectory.v1",
  "keyframes": [
    {
      "t": 0.0,
      "controls": {
        "size": -0.2,
        "distance": 0.0,
        "material": 0.0,
        "clarity": 0.2
      }
    },
    {
      "t": 1.0,
      "controls": {
        "size": 0.9,
        "distance": 0.5,
        "material": 0.3,
        "clarity": -0.1
      }
    }
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

Current implementation uses trajectory-conditioned segmented synthesis and overlap blending.

## Practical Guidance

- Start with small macro moves (`|value| <= 0.4`) before extreme values.
- Combine macros with explicit overrides (`--t60`, `--predelay-ms`) when exact targets are needed.
- Keep deterministic seeds during A/B work.
