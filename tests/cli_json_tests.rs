use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn generate_writes_metadata_and_analysis_json() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("gen.wav");
    let meta_path = dir.path().join("gen.meta.json");
    let analysis_path = dir.path().join("gen.analysis.json");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "dark steel bunker",
        "--seed",
        "99",
        "--duration",
        "0.5",
        "--t60",
        "0.8",
        "--output",
        ir_path.to_str().expect("utf8 path"),
        "--metadata-out",
        meta_path.to_str().expect("utf8 path"),
        "--json-analysis-out",
        analysis_path.to_str().expect("utf8 path"),
    ])
    .expect("cli parse should succeed");

    dispatch(cli).expect("generate command should succeed");

    let meta_text = std::fs::read_to_string(&meta_path).expect("metadata json should exist");
    let meta: Value = serde_json::from_str(&meta_text).expect("metadata should be valid json");
    assert_eq!(meta["schema_version"], "latent-ir.generation.v1");
    assert_eq!(meta["command"], "generate");
    assert_eq!(meta["seed"], 99);
    assert!(meta["descriptor"].is_object());
    assert!(meta["analysis"].is_object());

    let analysis_text =
        std::fs::read_to_string(&analysis_path).expect("analysis json should exist");
    let analysis: Value =
        serde_json::from_str(&analysis_text).expect("analysis should be valid json");
    assert_eq!(analysis["schema_version"], "latent-ir.analysis.v1");
    assert!(analysis["t60_s_est"].is_number() || analysis["t60_s_est"].is_null());
}

#[test]
fn analyze_json_output_includes_schema() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("probe.wav");
    let analysis_out = dir.path().join("analysis.json");

    let gen = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--preset",
        "frozen_plate",
        "--duration",
        "0.4",
        "--output",
        ir_path.to_str().expect("utf8 path"),
    ])
    .expect("generate parse should succeed");
    dispatch(gen).expect("generate should succeed");

    let analyze = Cli::try_parse_from([
        "latent-ir",
        "analyze",
        ir_path.to_str().expect("utf8 path"),
        "--json",
        "--output",
        analysis_out.to_str().expect("utf8 path"),
    ])
    .expect("analyze parse should succeed");
    dispatch(analyze).expect("analyze should succeed");

    let text = std::fs::read_to_string(&analysis_out).expect("analysis output should exist");
    let json: Value = serde_json::from_str(&text).expect("analysis output must be valid json");
    assert_eq!(json["schema_version"], "latent-ir.analysis.v1");
    assert!(json["duration_s"].is_number());
    assert!(json["spectral_centroid_hz"].is_number());
}

#[test]
fn generate_supports_foa_and_atmos_channel_formats() {
    let dir = tempdir().expect("tempdir");
    let foa_path = dir.path().join("foa.wav");
    let foa_meta = dir.path().join("foa.meta.json");
    let atmos_path = dir.path().join("atmos.wav");
    let atmos_meta = dir.path().join("atmos.meta.json");

    let foa = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "wide concrete silo",
        "--duration",
        "0.2",
        "--channels",
        "foa",
        "--output",
        foa_path.to_str().expect("utf8 path"),
        "--metadata-out",
        foa_meta.to_str().expect("utf8 path"),
    ])
    .expect("foa parse should succeed");
    dispatch(foa).expect("foa generate should succeed");

    let atmos = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "massive industrial chamber",
        "--duration",
        "0.2",
        "--channels",
        "7.2.4",
        "--output",
        atmos_path.to_str().expect("utf8 path"),
        "--metadata-out",
        atmos_meta.to_str().expect("utf8 path"),
    ])
    .expect("atmos parse should succeed");
    dispatch(atmos).expect("atmos generate should succeed");

    let foa_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&foa_meta).expect("foa metadata")).unwrap();
    assert_eq!(foa_json["channel_format"], "foa_ambix");
    assert_eq!(foa_json["spatial_encoding"], "ambisonic_foa_ambix");
    assert_eq!(
        foa_json["channel_labels"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0),
        4
    );
    assert_eq!(foa_json["analysis"]["channels"], 4);

    let atmos_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&atmos_meta).expect("atmos metadata"))
            .unwrap();
    assert_eq!(atmos_json["channel_format"], "atmos_7_2_4");
    assert_eq!(atmos_json["spatial_encoding"], "discrete");
    assert_eq!(
        atmos_json["channel_labels"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0),
        13
    );
    assert_eq!(atmos_json["analysis"]["channels"], 13);
}

#[test]
fn generate_supports_custom_layout_json_and_writes_channel_map() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("custom16.wav");
    let meta_path = dir.path().join("custom16.meta.json");
    let layout_path = dir.path().join("layout_16_0.json");
    let channel_map_path = dir.path().join("custom16.channels.json");
    let analysis_path = dir.path().join("custom16.analysis.json");

    let layout = r#"{
  "schema_version": "latent-ir.layout.v1",
  "layout_name": "custom_16_0_ring",
  "spatial_encoding": "discrete",
  "channels": [
    {"label":"C00","azimuth_deg":0,"elevation_deg":0},
    {"label":"C01","azimuth_deg":23,"elevation_deg":0},
    {"label":"C02","azimuth_deg":45,"elevation_deg":0},
    {"label":"C03","azimuth_deg":68,"elevation_deg":0},
    {"label":"C04","azimuth_deg":90,"elevation_deg":0},
    {"label":"C05","azimuth_deg":113,"elevation_deg":0},
    {"label":"C06","azimuth_deg":135,"elevation_deg":0},
    {"label":"C07","azimuth_deg":158,"elevation_deg":0},
    {"label":"C08","azimuth_deg":180,"elevation_deg":0},
    {"label":"C09","azimuth_deg":-158,"elevation_deg":0},
    {"label":"C10","azimuth_deg":-135,"elevation_deg":0},
    {"label":"C11","azimuth_deg":-113,"elevation_deg":0},
    {"label":"C12","azimuth_deg":-90,"elevation_deg":0},
    {"label":"C13","azimuth_deg":-68,"elevation_deg":0},
    {"label":"C14","azimuth_deg":-45,"elevation_deg":0},
    {"label":"C15","azimuth_deg":-23,"elevation_deg":0}
  ]
}"#;
    std::fs::write(&layout_path, layout).expect("write layout");

    let generate = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "large circular arena",
        "--duration",
        "0.2",
        "--channels",
        "custom",
        "--layout-json",
        layout_path.to_str().expect("utf8"),
        "--channel-map-out",
        channel_map_path.to_str().expect("utf8"),
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("generate parse should succeed");
    dispatch(generate).expect("generate should succeed");

    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).expect("metadata")).unwrap();
    assert_eq!(meta["channel_format"], "custom_16_0_ring");
    assert_eq!(meta["spatial_encoding"], "discrete");
    assert_eq!(meta["analysis"]["channels"], 16);
    assert_eq!(
        meta["channel_labels"].as_array().map(Vec::len).unwrap_or(0),
        16
    );
    assert_eq!(
        meta["channel_map_path"].as_str().unwrap_or_default(),
        channel_map_path.to_str().expect("utf8")
    );

    let map: Value =
        serde_json::from_str(&std::fs::read_to_string(&channel_map_path).expect("channel map"))
            .unwrap();
    assert_eq!(map["schema_version"], "latent-ir.channel-map.v1");
    assert_eq!(map["channels"].as_array().map(Vec::len).unwrap_or(0), 16);

    let analyze = Cli::try_parse_from([
        "latent-ir",
        "analyze",
        ir_path.to_str().expect("utf8"),
        "--channel-map",
        channel_map_path.to_str().expect("utf8"),
        "--json",
        "--output",
        analysis_path.to_str().expect("utf8"),
    ])
    .expect("analyze parse should succeed");
    dispatch(analyze).expect("analyze should succeed");

    let analysis: Value =
        serde_json::from_str(&std::fs::read_to_string(&analysis_path).expect("analysis report"))
            .unwrap();
    assert!(analysis["inter_channel_correlation_matrix"].is_array());
    assert!(analysis["front_energy_ratio"].is_number() || analysis["front_energy_ratio"].is_null());
}

#[test]
fn generate_custom_layout_requires_layout_json() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("invalid.wav");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "invalid custom test",
        "--channels",
        "custom",
        "--output",
        ir_path.to_str().expect("utf8"),
    ])
    .expect("parse should succeed");

    let err = dispatch(cli).expect_err("custom generation without layout should fail");
    assert!(err.to_string().contains("--channels custom requires --layout-json"));
}
