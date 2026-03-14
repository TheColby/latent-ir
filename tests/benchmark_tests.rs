use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use latent_ir::core::descriptors::DescriptorSet;
use serde_json::{json, Value};
use tempfile::tempdir;

#[test]
fn benchmark_run_writes_report() {
    let dir = tempdir().expect("tempdir");
    let dataset = dir.path().join("benchmark_dataset.json");
    let report = dir.path().join("benchmark_report.json");

    let mut d1 = DescriptorSet::default();
    d1.time.t60 = 8.0;
    let mut d2 = DescriptorSet::default();
    d2.time.t60 = 1.2;

    let ds = json!({
      "schema_version": "latent-ir.benchmark.dataset.v1",
      "samples": [
        {"id": "a", "prompt": "dark cathedral", "target_descriptor": d1},
        {"id": "b", "prompt": "intimate bright room", "target_descriptor": d2}
      ]
    });
    std::fs::write(&dataset, serde_json::to_string_pretty(&ds).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "benchmark",
        "run",
        "--dataset",
        dataset.to_str().unwrap(),
        "--output",
        report.to_str().unwrap(),
        "--repeats",
        "2",
    ])
    .unwrap();
    dispatch(cli).expect("benchmark run should pass");

    let v: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(v["schema_version"], "latent-ir.benchmark.v1");
    assert!(v["summary"]["total_score"].is_number());
}

#[test]
fn benchmark_check_detects_regression() {
    let dir = tempdir().expect("tempdir");
    let baseline = dir.path().join("baseline.json");
    let report = dir.path().join("report.json");

    let base = json!({
      "schema_version": "latent-ir.benchmark.v1",
      "generated_at_utc": "2026-03-10T00:00:00Z",
      "dataset_path": "x",
      "sample_count": 1,
      "repeats": 1,
      "objective": {"descriptor_mae": 0.1, "descriptor_rmse": 0.1, "analysis_mae": 0.1, "analysis_rmse": 0.1},
      "speed": {"encode_ms_avg": 1.0, "generate_ms_avg": 1.0, "analyze_ms_avg": 1.0, "total_ms_avg": 3.0},
      "stability": {"descriptor_std_avg": 0.0, "analysis_std_avg": 0.0},
      "perceptual_proxy": {"proxy_mae": 0.1, "per_proxy_mae": {"clarity": 0.1, "brightness": 0.1, "spaciousness": 0.1, "distance": 0.1}},
      "summary": {"objective_score": 0.2, "speed_score": 0.003, "stability_score": 0.0, "perceptual_score": 0.1, "total_score": 0.303}
    });
    let bad = json!({
      "schema_version": "latent-ir.benchmark.v1",
      "generated_at_utc": "2026-03-10T00:00:00Z",
      "dataset_path": "x",
      "sample_count": 1,
      "repeats": 1,
      "objective": {"descriptor_mae": 0.2, "descriptor_rmse": 0.2, "analysis_mae": 0.2, "analysis_rmse": 0.2},
      "speed": {"encode_ms_avg": 1.0, "generate_ms_avg": 1.0, "analyze_ms_avg": 1.0, "total_ms_avg": 3.0},
      "stability": {"descriptor_std_avg": 0.0, "analysis_std_avg": 0.0},
      "perceptual_proxy": {"proxy_mae": 0.2, "per_proxy_mae": {"clarity": 0.2, "brightness": 0.2, "spaciousness": 0.2, "distance": 0.2}},
      "summary": {"objective_score": 0.4, "speed_score": 0.003, "stability_score": 0.0, "perceptual_score": 0.2, "total_score": 0.603}
    });

    std::fs::write(&baseline, serde_json::to_string_pretty(&base).unwrap()).unwrap();
    std::fs::write(&report, serde_json::to_string_pretty(&bad).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "benchmark",
        "check",
        "--report",
        report.to_str().unwrap(),
        "--baseline",
        baseline.to_str().unwrap(),
        "--max-regression",
        "0.05",
    ])
    .unwrap();

    assert!(dispatch(cli).is_err());
}

#[test]
fn benchmark_trend_writes_dashboard_outputs() {
    let dir = tempdir().expect("tempdir");
    let r1 = dir.path().join("r1.json");
    let r2 = dir.path().join("r2.json");
    let out_md = dir.path().join("trend.md");

    let base = json!({
      "schema_version": "latent-ir.benchmark.v1",
      "generated_at_utc": "2026-03-10T00:00:00Z",
      "dataset_path": "x",
      "sample_count": 1,
      "repeats": 1,
      "objective": {"descriptor_mae": 0.1, "descriptor_rmse": 0.1, "analysis_mae": 0.1, "analysis_rmse": 0.1},
      "speed": {"encode_ms_avg": 1.0, "generate_ms_avg": 1.0, "analyze_ms_avg": 1.0, "total_ms_avg": 3.0},
      "stability": {"descriptor_std_avg": 0.0, "analysis_std_avg": 0.0},
      "perceptual_proxy": {"proxy_mae": 0.1, "per_proxy_mae": {"clarity": 0.1, "brightness": 0.1, "spaciousness": 0.1, "distance": 0.1}},
      "summary": {"objective_score": 0.2, "speed_score": 0.003, "stability_score": 0.0, "perceptual_score": 0.1, "total_score": 0.303}
    });
    let next = json!({
      "schema_version": "latent-ir.benchmark.v1",
      "generated_at_utc": "2026-03-11T00:00:00Z",
      "dataset_path": "x",
      "sample_count": 1,
      "repeats": 1,
      "objective": {"descriptor_mae": 0.08, "descriptor_rmse": 0.08, "analysis_mae": 0.08, "analysis_rmse": 0.08},
      "speed": {"encode_ms_avg": 1.1, "generate_ms_avg": 1.0, "analyze_ms_avg": 1.0, "total_ms_avg": 3.1},
      "stability": {"descriptor_std_avg": 0.0, "analysis_std_avg": 0.0},
      "perceptual_proxy": {"proxy_mae": 0.09, "per_proxy_mae": {"clarity": 0.09, "brightness": 0.09, "spaciousness": 0.09, "distance": 0.09}},
      "summary": {"objective_score": 0.16, "speed_score": 0.0031, "stability_score": 0.0, "perceptual_score": 0.09, "total_score": 0.2531}
    });
    std::fs::write(&r1, serde_json::to_string_pretty(&base).unwrap()).unwrap();
    std::fs::write(&r2, serde_json::to_string_pretty(&next).unwrap()).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "benchmark",
        "trend",
        "--reports",
        r1.to_str().unwrap(),
        "--reports",
        r2.to_str().unwrap(),
        "--output",
        out_md.to_str().unwrap(),
    ])
    .unwrap();
    dispatch(cli).expect("benchmark trend should pass");

    assert!(out_md.exists());
    let out_json = out_md.with_extension("json");
    assert!(out_json.exists());
    let md = std::fs::read_to_string(&out_md).unwrap();
    assert!(md.contains("Benchmark Trend Dashboard"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(out_json).unwrap()).unwrap();
    assert_eq!(v["schema_version"], "latent-ir.benchmark.trend.v1");
    assert_eq!(v["summary"]["point_count"], 2);
}
