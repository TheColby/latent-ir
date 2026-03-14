# Launch Readiness

This document is a practical pre-launch checklist for public announcements.

## Common Public Questions and Ready Answers

1. Where are the audio demos?
- Provide deterministic demo artifacts, not screenshots.
- Use `./scripts/generate_demo_pack.sh`.

2. Is this really ML?
- Be explicit: generation is DSP-first.
- Learned components currently affect conditioning, not opaque end-to-end IR generation.

3. How trustworthy are RT/EDT metrics?
- Present them as deterministic engineering estimates.
- Do not market them as certified architectural acoustics metrology.
- Include confidence context (`decay_db_span`, `t60_confidence`, `edt_confidence`) when sharing metrics publicly.

4. What spatial formats are truly supported?
- Built-ins: `mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`.
- Custom layouts: `--channels custom --layout-json ...`.
- Object-based Atmos rendering is currently out of scope.

5. How does this fit production workflows?
- Generate variants
- Analyze/filter candidates
- Morph finalists
- Render stems offline
- Audition in DAW/convolution host

6. Can it handle long multichannel jobs?
- Use `--engine auto` or `--engine fft-streaming`.
- Engine choice is reported in console.

7. What about sample-rate mismatches?
- Use `--auto-resample --resample-mode linear|cubic` for `render`/`morph` when needed.
- Keep strict mode by default in automated pipelines when reproducibility requires exact inputs.

8. Conditioning is a black box.
- Use `generate --explain-conditioning` and include metadata deltas in release artifacts.
- Mention whether semantic rules, learned text model, and/or learned audio model were active.

8. PATH/install issues?
- Use `./scripts/install_local.sh`.
- Ensure `$HOME/.cargo/bin` (or custom `$CARGO_HOME/bin`) is on `PATH`.

## Launch Artifact Checklist

- 2-4 generated IR WAV examples (contrasting acoustic intent)
- JSON analysis sidecar per IR
- one morph example
- one render A/B clip per IR (same dry source)
- exact command lines used

## Suggested Scope-Honesty Blurb

`latent-ir is a deterministic DSP-first IR synthesis toolkit with hybrid conditioning hooks. Current metrics are engineering estimates for reproducible workflow comparison.`

Use this wording (or equivalent) to preempt avoidable credibility friction.
