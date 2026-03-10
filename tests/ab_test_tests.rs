use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn ab_test_generates_comparison_artifacts() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("ab");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "ab-test",
        "--prompt",
        "massive colossal grain silo made of poured concrete with rt60 27 seconds",
        "--industrial-text-model",
        "examples/models/text_encoder_v1.json",
        "--output-dir",
        out_dir.to_str().expect("utf8"),
        "--t60",
        "27",
        "--macro-size",
        "1.0",
        "--macro-distance",
        "0.8",
        "--markdown",
    ])
    .expect("cli parse");

    dispatch(cli).expect("ab-test should run");

    let report_path = out_dir.join("ab_test_report.json");
    assert!(report_path.exists());

    let v: Value = serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(v["schema_version"], "latent-ir.ab-test.v1");
    assert!(out_dir.join("industrial.wav").exists());
    assert!(out_dir.join("baseline.wav").exists());
    assert!(out_dir.join("industrial.analysis.json").exists());
    assert!(out_dir.join("baseline.analysis.json").exists());
    assert!(out_dir.join("ab_test_report.md").exists());
    assert!(v["delta"]["t60_s_est"].is_number());
}
