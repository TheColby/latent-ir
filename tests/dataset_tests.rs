use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn dataset_synthesize_writes_manifest_and_training_exports() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("dataset");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "dataset",
        "synthesize",
        "--out-dir",
        out_dir.to_str().expect("utf8"),
        "--count",
        "4",
        "--seed",
        "88",
        "--sample-rate",
        "8000",
        "--channels",
        "mono",
        "--duration-min",
        "0.12",
        "--duration-max",
        "0.18",
        "--t60-min",
        "0.2",
        "--t60-max",
        "0.4",
        "--predelay-max-ms",
        "8",
        "--tail-fade-ms",
        "20",
        "--export-training-json",
    ])
    .expect("parse");

    dispatch(cli).expect("dataset synth should succeed");

    let manifest_path = out_dir.join("manifest.dataset.json");
    let text_train_path = out_dir.join("training_text.json");
    let audio_train_path = out_dir.join("training_audio.json");
    assert!(manifest_path.exists());
    assert!(text_train_path.exists());
    assert!(audio_train_path.exists());

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest text"))
            .expect("manifest json");
    assert_eq!(manifest["schema_version"], "latent-ir.dataset.v1");

    let succeeded = manifest["config"]["count_succeeded"].as_u64().unwrap_or(0) as usize;
    assert!(succeeded > 0);
    let records = manifest["records"].as_array().expect("records array");
    assert_eq!(records.len(), succeeded);

    let text_train: Value =
        serde_json::from_str(&std::fs::read_to_string(&text_train_path).expect("text train"))
            .expect("text train json");
    let audio_train: Value =
        serde_json::from_str(&std::fs::read_to_string(&audio_train_path).expect("audio train"))
            .expect("audio train json");
    assert_eq!(text_train.as_array().map(Vec::len).unwrap_or(0), succeeded);
    assert_eq!(audio_train.as_array().map(Vec::len).unwrap_or(0), succeeded);
}

#[test]
fn dataset_synthesize_uses_prompt_bank_file() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("dataset");
    let prompt_bank = dir.path().join("prompts.json");
    std::fs::write(
        &prompt_bank,
        r#"[
  "industrial ventilation shaft",
  "small marble chamber"
]"#,
    )
    .expect("write prompt bank");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "dataset",
        "synthesize",
        "--out-dir",
        out_dir.to_str().expect("utf8"),
        "--count",
        "3",
        "--seed",
        "99",
        "--sample-rate",
        "8000",
        "--channels",
        "mono",
        "--duration-min",
        "0.1",
        "--duration-max",
        "0.12",
        "--t60-min",
        "0.2",
        "--t60-max",
        "0.3",
        "--predelay-max-ms",
        "5",
        "--preset-mix",
        "0.0",
        "--prompt-bank-json",
        prompt_bank.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("dataset synth should succeed");

    let manifest_path = out_dir.join("manifest.dataset.json");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest text"))
            .expect("manifest json");
    let allowed = ["industrial ventilation shaft", "small marble chamber"];
    for rec in manifest["records"].as_array().expect("records") {
        let prompt = rec["prompt"].as_str().unwrap_or_default();
        assert!(allowed.contains(&prompt));
    }
}
