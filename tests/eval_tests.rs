use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use latent_ir::core::descriptors::DescriptorSet;
use latent_ir::core::util;
use serde_json::{json, Value};
use tempfile::tempdir;

#[test]
fn eval_text_writes_baseline_report() {
    let dir = tempdir().expect("tempdir");
    let dataset = dir.path().join("text_dataset.json");
    let model = dir.path().join("text_model.json");
    let baseline = dir.path().join("text_baseline.json");

    let mut a = DescriptorSet::default();
    a.time.t60 = 7.0;
    let mut b = DescriptorSet::default();
    b.time.t60 = 1.0;

    let ds = json!([
        {"prompt": "dark cathedral", "descriptor": a},
        {"prompt": "bright room", "descriptor": b}
    ]);
    std::fs::write(&dataset, serde_json::to_string_pretty(&ds).unwrap()).unwrap();

    let train = Cli::try_parse_from([
        "latent-ir",
        "train-encoder",
        "text",
        "--dataset",
        dataset.to_str().unwrap(),
        "--output",
        model.to_str().unwrap(),
        "--epochs",
        "80",
    ])
    .unwrap();
    dispatch(train).unwrap();

    let eval = Cli::try_parse_from([
        "latent-ir",
        "eval",
        "text",
        "--dataset",
        dataset.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--output",
        baseline.to_str().unwrap(),
    ])
    .unwrap();
    dispatch(eval).unwrap();

    let v: Value = serde_json::from_str(&std::fs::read_to_string(&baseline).unwrap()).unwrap();
    assert_eq!(v["schema_version"], "latent-ir.eval.baseline.v1");
    assert_eq!(v["command"], "eval text");
    assert!(v["descriptor_metrics"]["mae"].is_number());
    assert!(v["analysis_metrics"]["mae"].is_number());
}

#[test]
fn eval_audio_writes_baseline_report() {
    let dir = tempdir().expect("tempdir");
    let dataset = dir.path().join("audio_dataset.json");
    let model = dir.path().join("audio_model.json");
    let baseline = dir.path().join("audio_baseline.json");

    let sr = 48_000;
    let ir1 = dir.path().join("ir1.wav");
    let ir2 = dir.path().join("ir2.wav");

    let mut s1 = vec![0.0f32; 10_000];
    s1[120] = 1.0;
    for i in 121..s1.len() {
        let t = (i - 120) as f32 / sr as f32;
        s1[i] = (-(t / 0.7)).exp() * 0.09;
    }
    util::audio::write_wav_f32(&ir1, sr, &[s1]).unwrap();

    let mut s2 = vec![0.0f32; 10_000];
    s2[40] = 1.0;
    for i in 41..s2.len() {
        let t = (i - 40) as f32 / sr as f32;
        s2[i] = (-(t / 0.2)).exp() * 0.12;
    }
    util::audio::write_wav_f32(&ir2, sr, &[s2]).unwrap();

    let mut d1 = DescriptorSet::default();
    d1.time.t60 = 6.0;
    let mut d2 = DescriptorSet::default();
    d2.time.t60 = 1.1;

    let ds = json!([
        {"audio_path": "ir1.wav", "descriptor": d1},
        {"audio_path": "ir2.wav", "descriptor": d2}
    ]);
    std::fs::write(&dataset, serde_json::to_string_pretty(&ds).unwrap()).unwrap();

    let train = Cli::try_parse_from([
        "latent-ir",
        "train-encoder",
        "audio",
        "--dataset",
        dataset.to_str().unwrap(),
        "--output",
        model.to_str().unwrap(),
        "--epochs",
        "100",
    ])
    .unwrap();
    dispatch(train).unwrap();

    let eval = Cli::try_parse_from([
        "latent-ir",
        "eval",
        "audio",
        "--dataset",
        dataset.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--output",
        baseline.to_str().unwrap(),
    ])
    .unwrap();
    dispatch(eval).unwrap();

    let v: Value = serde_json::from_str(&std::fs::read_to_string(&baseline).unwrap()).unwrap();
    assert_eq!(v["schema_version"], "latent-ir.eval.baseline.v1");
    assert_eq!(v["command"], "eval audio");
    assert!(v["sample_count"].as_u64().unwrap_or(0) >= 2);
    assert!(v["analysis_metrics"]["per_metric_mae"].is_object());
}
