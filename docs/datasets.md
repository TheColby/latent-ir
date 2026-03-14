# Dataset Workflows

`latent-ir` includes a dataset synthesis pipeline for AI research and augmentation workflows.

Primary command:

```bash
latent-ir dataset synthesize --out-dir out/research_dataset --count 512
```

## What It Produces

By default, each synthesized sample writes:
- `ir/*.wav`
- `metadata/*.json` (`latent-ir.generation.v1`)
- `analysis/*.json` (`latent-ir.analysis.v1`)
- `channel_maps/*.channels.json`

Dataset-level artifact:
- `manifest.dataset.json` (`latent-ir.dataset.v1`)

Optional exports (`--export-training-json`):
- `training_text.json` for prompt->descriptor training
- `training_audio.json` for audio->descriptor training

These export formats are directly compatible with:
- `latent-ir train-encoder text --dataset ...`
- `latent-ir train-encoder audio --dataset ...`

## Typical Research Use Cases

1. Bootstrap prompt-conditioned encoder datasets
- sample broad prompt banks with preset mixing and descriptor jitter
- export `training_text.json`

2. Bootstrap audio-conditioned descriptor datasets
- treat generated IR audio as labeled descriptor examples
- export `training_audio.json`

3. Controlled ablation corpora
- fix `--seed`
- constrain `--duration-min/max` and `--t60-min/max`
- compare model behavior under stable distributions

4. Quality-filtered corpora
- enable `--quality-gate --quality-profile launch|strict`
- use manifest summary + warnings for triage

## Important Flags

- `--prompt-bank-json <path>`: JSON string array of prompts
- `--preset-mix <0..1>`: probability of applying a built-in preset before prompt conditioning
- `--jitter <0..1>`: random variation strength for selected descriptor controls
- `--tail-fade-ms <ms>`: optional forced zero-end taper
- `--quality-gate --quality-profile <lenient|launch|strict>`: per-sample quality filter context
- `--export-training-json`: emit train-encoder-ready files

## Example

```bash
latent-ir dataset synthesize \
  --out-dir out/industrial_corpus \
  --count 1024 \
  --seed 2027 \
  --channels stereo \
  --duration-min 0.8 --duration-max 6.0 \
  --t60-min 0.5 --t60-max 10.0 \
  --predelay-max-ms 80 \
  --jitter 0.25 \
  --preset-mix 0.45 \
  --quality-gate --quality-profile launch \
  --tail-fade-ms 25 \
  --export-training-json
```

## Reproducibility Notes

- The command is deterministic for a given `--seed` and config.
- Each sample records:
  - synthesis seed
  - descriptor values
  - analysis report
  - reproducibility hashes in metadata
- The dataset manifest includes aggregate summary metrics and pass/fail counts.
