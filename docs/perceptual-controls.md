# Perceptual Controls

`latent-ir generate` exposes macro controls for fast, human-readable descriptor steering.

## Macro Control Range

Each macro uses `[-1, 1]`.

- `--macro-size`: perceived space scale
- `--macro-distance`: perceived source/listener distance
- `--macro-material`: perceived hardness/brightness
- `--macro-clarity`: perceived clarity vs smear

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

`--macro-trajectory` accepts keyframes with normalized time.

Schema:
- `latent-ir.macro-trajectory.v1`

Example:

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

Run:

```bash
cargo run -- generate \
  --prompt "impossible tunnel" \
  --macro-trajectory examples/macro_trajectory_rise.json \
  --output out/trajectory_ir.wav
```

## Practical Use

- Keep macro values moderate (`|value| <= 0.4`) before pushing extremes.
- Combine macros with hard constraints (`--t60`, `--predelay-ms`) when exact targets matter.
- Keep seed fixed during A/B comparisons.
