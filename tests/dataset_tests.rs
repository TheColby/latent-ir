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

#[test]
fn dataset_split_writes_hash_locked_manifest_and_split_training_exports() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("dataset");

    let synth = Cli::try_parse_from([
        "latent-ir",
        "dataset",
        "synthesize",
        "--out-dir",
        out_dir.to_str().expect("utf8"),
        "--count",
        "10",
        "--seed",
        "123",
        "--sample-rate",
        "8000",
        "--channels",
        "mono",
        "--duration-min",
        "0.1",
        "--duration-max",
        "0.14",
        "--t60-min",
        "0.2",
        "--t60-max",
        "0.35",
        "--predelay-max-ms",
        "6",
    ])
    .expect("parse");
    dispatch(synth).expect("dataset synth should succeed");

    let split_path = out_dir.join("split.dataset.json");
    let split = Cli::try_parse_from([
        "latent-ir",
        "dataset",
        "split",
        "--manifest",
        out_dir
            .join("manifest.dataset.json")
            .to_str()
            .expect("utf8 manifest"),
        "--output",
        split_path.to_str().expect("utf8 split"),
        "--seed",
        "2027",
        "--train-ratio",
        "0.7",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.1",
        "--lock-hashes",
        "--emit-training-json",
    ])
    .expect("parse split");
    dispatch(split).expect("dataset split should succeed");

    assert!(split_path.exists());
    let split_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&split_path).expect("split text"))
            .expect("split json");
    assert_eq!(split_json["schema_version"], "latent-ir.dataset-split.v1");
    assert_eq!(split_json["hash_locked"], true);

    let total = split_json["counts"]["total"].as_u64().unwrap_or(0);
    let train = split_json["counts"]["train"].as_u64().unwrap_or(0);
    let val = split_json["counts"]["val"].as_u64().unwrap_or(0);
    let test = split_json["counts"]["test"].as_u64().unwrap_or(0);
    assert_eq!(total, train + val + test);
    assert!(train > 0);
    assert!(split_json["train"][0]["ir_sha256"].is_string());

    assert!(out_dir.join("train_text.json").exists());
    assert!(out_dir.join("train_audio.json").exists());
    assert!(out_dir.join("val_text.json").exists());
    assert!(out_dir.join("test_audio.json").exists());
}
