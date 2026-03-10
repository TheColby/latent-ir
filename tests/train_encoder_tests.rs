use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use latent_ir::core::conditioning::{
    AudioEncoder, LearnedAudioEncoder, LearnedTextEncoder, TextEncoder,
};
use latent_ir::core::descriptors::DescriptorSet;
use latent_ir::core::util;
use serde_json::json;
use tempfile::tempdir;

#[test]
fn train_text_encoder_writes_usable_model() {
    let dir = tempdir().expect("tempdir");
    let dataset = dir.path().join("text_dataset.json");
    let out = dir.path().join("text_model.json");

    let mut a = DescriptorSet::default();
    a.time.t60 = 8.0;
    a.spectral.brightness = 0.3;

    let mut b = DescriptorSet::default();
    b.time.t60 = 1.2;
    b.spectral.brightness = 0.9;

    let ds = json!([
        {"prompt": "dark stone cathedral", "descriptor": a},
        {"prompt": "bright steel plate", "descriptor": b}
    ]);
    std::fs::write(&dataset, serde_json::to_string_pretty(&ds).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "train-encoder",
        "text",
        "--dataset",
        dataset.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--epochs",
        "80",
        "--lr",
        "0.1",
    ])
    .expect("parse");
    dispatch(cli).expect("train text");

    let model = LearnedTextEncoder::from_json_file(&out).expect("load model");
    let delta = model
        .infer_delta_from_prompt("dark cathedral")
        .expect("infer");
    assert!(delta.t60.abs() > 0.01);
}

#[test]
fn train_audio_encoder_writes_usable_model() {
    let dir = tempdir().expect("tempdir");
    let dataset = dir.path().join("audio_dataset.json");
    let out = dir.path().join("audio_model.json");

    let sr = 48_000;
    let ir1 = dir.path().join("ir1.wav");
    let ir2 = dir.path().join("ir2.wav");

    let mut s1 = vec![0.0f32; 12_000];
    s1[180] = 1.0;
    for i in 181..s1.len() {
        let t = (i - 180) as f32 / sr as f32;
        s1[i] = (-(t / 0.9)).exp() * 0.08;
    }
    util::audio::write_wav_f32(&ir1, sr, &[s1]).unwrap();

    let mut s2 = vec![0.0f32; 12_000];
    s2[40] = 1.0;
    for i in 41..s2.len() {
        let t = (i - 40) as f32 / sr as f32;
        s2[i] = (-(t / 0.2)).exp() * 0.12;
    }
    util::audio::write_wav_f32(&ir2, sr, &[s2]).unwrap();

    let mut d1 = DescriptorSet::default();
    d1.time.t60 = 7.5;
    let mut d2 = DescriptorSet::default();
    d2.time.t60 = 0.9;

    let ds = json!([
        {"audio_path": "ir1.wav", "descriptor": d1},
        {"audio_path": "ir2.wav", "descriptor": d2}
    ]);
    std::fs::write(&dataset, serde_json::to_string_pretty(&ds).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "train-encoder",
        "audio",
        "--dataset",
        dataset.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--epochs",
        "120",
        "--lr",
        "0.05",
    ])
    .expect("parse");
    dispatch(cli).expect("train audio");

    let model = LearnedAudioEncoder::from_json_file(&out).expect("load model");
    let probe = util::audio::read_wav_f32(&ir1).unwrap();
    let delta = model
        .infer_delta_from_audio(&probe.channels, probe.sample_rate)
        .expect("infer");
    assert!(delta.t60.abs() > 0.01);
}
