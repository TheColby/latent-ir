# Dataset Workflows

`latent-ir` includes a dataset synthesis pipeline for AI research and augmentation workflows.

Primary command:

```bash
latent-ir dataset synthesize --out-dir out/research_dataset --count 512
```

Companion split command:

```bash
latent-ir dataset split \
  --manifest out/research_dataset/manifest.dataset.json \
  --output out/research_dataset/split.dataset.json \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --lock-hashes
```

Companion verify command:

```bash
latent-ir dataset verify \
  --split-manifest out/research_dataset/split.dataset.json \
  --fail-on-prompt-overlap \
  --output out/research_dataset/verify.dataset.json
```

## What It Produces

By default, each synthesized sample writes:
- `ir/*.wav`
- `metadata/*.json` (`latent-ir.generation.v1`)
- `analysis/*.json` (`latent-ir.analysis.v1`)
- `channel_maps/*.channels.json`

Dataset-level artifact:
- `manifest.dataset.json` (`latent-ir.dataset.v1`)
- `split.dataset.json` (`latent-ir.dataset-split.v1`)
- `verify.dataset.json` (`latent-ir.dataset-verify.v1`, optional)

Optional exports (`--export-training-json`):
- `training_text.json` for prompt->descriptor training
- `training_audio.json` for audio->descriptor training

Optional split exports (`dataset split --emit-training-json`):
- `train_text.json`, `val_text.json`, `test_text.json`
- `train_audio.json`, `val_audio.json`, `test_audio.json`

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

5. Hash-locked split reproducibility
- run `dataset split --lock-hashes`
- split records include per-sample metadata hashes (`ir_sha256`, `descriptor_sha256`, `channel_map_sha256`)

6. Split integrity / leakage gates
- run `dataset verify --split-manifest ...`
- checks missing files, hash mismatches, split ID overlap, and prompt overlap
- add `--fail-on-prompt-overlap` to make prompt leakage release-blocking

## Important Flags

- `--prompt-bank-json <path>`: JSON string array of prompts
- `--preset-mix <0..1>`: probability of applying a built-in preset before prompt conditioning
- `--jitter <0..1>`: random variation strength for selected descriptor controls
- `--tail-fade-ms <ms>`: optional forced zero-end taper
- `--quality-gate --quality-profile <lenient|launch|strict>`: per-sample quality filter context
- `--export-training-json`: emit train-encoder-ready files
- `dataset split --lock-hashes`: embed integrity hashes in split records
- `dataset split --emit-training-json`: emit split-specific training datasets
- `dataset verify --split-manifest`: run integrity/leakage checks
- `dataset verify --fail-on-prompt-overlap`: fail when prompt overlap exists across splits
- `dataset verify --output`: write machine-readable verification report

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
- The split verification report captures integrity and leakage status for CI/release gates.
