# Launch Readiness Notes

This document is a practical checklist for public launch posts (for example LinkedIn, X, forum announcements) and the top technical questions people usually ask.

## Expected Questions and Project-Backed Answers

1. Where are the demos?
- Answer with reproducible assets, not screenshots.
- Use `./scripts/generate_demo_pack.sh` to generate `out/demos/*.wav` plus analysis JSON.

2. Is this actually ML?
- Current production path is procedural DSP generation.
- ML in v0/v0.3 is conditioning support (semantic rules + lightweight learned encoders), not a hidden large model generator.
- State this directly in launch copy.

3. How trustworthy are RT60/EDT numbers?
- Metrics are deterministic engineering estimates intended for comparative workflow use.
- They are not standards-certified room-acoustics metrology.
- CLI output now prints this caveat in both `generate` and `analyze`.

4. What spatial formats are truly supported?
- Built-ins: `mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`.
- Custom: `--channels custom --layout-json ...` with cartesian and/or polar metadata.
- Object-based Atmos rendering is explicitly out of scope for current versions.

5. How does this fit production?
- Recommend a reproducible pipeline:
  - `generate` variants
  - `analyze` and score/filter
  - `morph` candidates
  - `render` stems offline
  - audition in DAW

6. Does this scale to large jobs?
- Render `--engine auto` now selects `direct`, `fft-partitioned`, or `fft-streaming` based on workload/IR size and prints the decision.
- For long-form multichannel, use `fft-streaming`.
- Mixed sample-rate assets can be reconciled with `--auto-resample` in `render` and `morph`.

7. Install issues (`latent-ir: command not found`)
- Use `./scripts/install_local.sh`.
- Ensure cargo bin directory is on `PATH` (`$HOME/.cargo/bin` unless custom `CARGO_HOME`).

## Suggested Launch Artifact Set

- 2 to 4 deterministic IR WAV files (different acoustic intents).
- 1 morph example WAV.
- JSON analysis for each WAV.
- 1 short A/B rendered clip per IR using the same dry source.
- Exact command lines used to generate each artifact.

## Scope Honesty Snippet

Use wording similar to:

`latent-ir is currently a deterministic DSP-first IR synthesis system with hybrid conditioning hooks. Current metrics are engineering estimates for reproducible workflow comparisons.`

This removes most avoidable credibility friction.
