use clap::Parser;
use latent_ir::cli::Cli;
use latent_ir::commands::dispatch;
use latent_ir::core::util;
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
        "--explain-conditioning",
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
    assert!(meta["replay_command"].is_string());
    assert!(meta["descriptor"].is_object());
    assert!(meta["analysis"].is_object());
    assert!(meta["conditioning"]["combined_delta"].is_object());
    assert!(meta["ir_sha256"].as_str().unwrap_or("").len() >= 64);
    assert!(meta["descriptor_sha256"].as_str().unwrap_or("").len() >= 64);
    assert!(meta["channel_map_sha256"].as_str().unwrap_or("").len() >= 64);

    let analysis_text =
        std::fs::read_to_string(&analysis_path).expect("analysis json should exist");
    let analysis: Value =
        serde_json::from_str(&analysis_text).expect("analysis should be valid json");
    assert_eq!(analysis["schema_version"], "latent-ir.analysis.v1");
    assert!(analysis["t60_s_est"].is_number() || analysis["t60_s_est"].is_null());
    assert!(analysis["arrival_spread_ms"].is_number() || analysis["arrival_spread_ms"].is_null());
    assert!(analysis["itd_01_ms"].is_number() || analysis["itd_01_ms"].is_null());
    assert!(analysis["iacc_early_01"].is_number() || analysis["iacc_early_01"].is_null());
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
    assert!(err
        .to_string()
        .contains("--channels custom requires --layout-json"));
}

#[test]
fn generate_supports_cartesian_only_custom_layout() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("cart.wav");
    let meta_path = dir.path().join("cart.meta.json");
    let map_path = dir.path().join("cart.channels.json");
    let layout_path = dir.path().join("layout_cartesian_only.json");

    let layout = r#"{
  "schema_version": "latent-ir.layout.v1",
  "layout_name": "quad_cartesian_only",
  "spatial_encoding": "discrete",
  "channels": [
    {"label":"F","position_m":{"x":0.0,"y":2.0,"z":0.0}},
    {"label":"R","position_m":{"x":2.0,"y":0.0,"z":0.0}},
    {"label":"B","position_m":{"x":0.0,"y":-2.0,"z":0.0}},
    {"label":"L","position_m":{"x":-2.0,"y":0.0,"z":0.0}}
  ]
}"#;
    std::fs::write(&layout_path, layout).expect("write layout");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "compact industrial room",
        "--duration",
        "0.2",
        "--channels",
        "custom",
        "--layout-json",
        layout_path.to_str().expect("utf8"),
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
        "--channel-map-out",
        map_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("generate should succeed");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["channel_format"], "quad_cartesian_only");
    assert_eq!(meta["analysis"]["channels"], 4);

    let map: Value = serde_json::from_str(&std::fs::read_to_string(&map_path).unwrap()).unwrap();
    let channels = map["channels"].as_array().expect("channels array");
    assert_eq!(channels.len(), 4);

    let az_f = channels[0]["azimuth_deg"].as_f64().unwrap() as f32;
    let az_r = channels[1]["azimuth_deg"].as_f64().unwrap() as f32;
    let az_b = channels[2]["azimuth_deg"].as_f64().unwrap() as f32;
    let az_l = channels[3]["azimuth_deg"].as_f64().unwrap() as f32;

    assert!((az_f - 0.0).abs() < 0.5);
    assert!((az_r - 90.0).abs() < 0.5);
    assert!((az_b - 180.0).abs() < 0.5 || (az_b + 180.0).abs() < 0.5);
    assert!((az_l + 90.0).abs() < 0.5);
    assert!(channels
        .iter()
        .all(|c| c["position_m"].is_object() && c["elevation_deg"].is_number()));
}

#[test]
fn generate_allows_layout_json_without_explicit_channels_flag() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("layout_only.wav");
    let meta_path = dir.path().join("layout_only.meta.json");
    let layout_path = dir.path().join("layout.json");

    let layout = r#"{
  "schema_version": "latent-ir.layout.v1",
  "layout_name": "layout_only_quad",
  "spatial_encoding": "discrete",
  "channels": [
    {"label":"F","azimuth_deg":0,"elevation_deg":0},
    {"label":"R","azimuth_deg":180,"elevation_deg":0},
    {"label":"L","azimuth_deg":-90,"elevation_deg":0},
    {"label":"Rt","azimuth_deg":90,"elevation_deg":0}
  ]
}"#;
    std::fs::write(&layout_path, layout).expect("write layout");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "layout-only test",
        "--layout-json",
        layout_path.to_str().expect("utf8"),
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("parse");
    dispatch(cli).expect("generate should succeed");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["channel_format"], "layout_only_quad");
    assert_eq!(meta["analysis"]["channels"], 4);
}

#[test]
fn generate_rejects_inconsistent_polar_and_cartesian_layout() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("bad.wav");
    let layout_path = dir.path().join("layout_bad.json");

    let bad_layout = r#"{
  "schema_version": "latent-ir.layout.v1",
  "layout_name": "bad_layout",
  "spatial_encoding": "discrete",
  "channels": [
    {"label":"C0","azimuth_deg":0,"elevation_deg":0,"position_m":{"x":1.0,"y":0.0,"z":0.0}}
  ]
}"#;
    std::fs::write(&layout_path, bad_layout).expect("write layout");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "bad geometry test",
        "--channels",
        "custom",
        "--layout-json",
        layout_path.to_str().expect("utf8"),
        "--output",
        ir_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    let err = dispatch(cli).expect_err("inconsistent layout should fail");
    assert!(err
        .to_string()
        .contains("inconsistent polar/cartesian geometry"));
}

#[test]
fn generate_accepts_virtual_source_and_listener_positions() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("pos.wav");
    let meta_path = dir.path().join("pos.meta.json");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "industrial hangar",
        "--duration",
        "0.2",
        "--source-x-m",
        "2.0",
        "--source-y-m",
        "9.0",
        "--source-z-m",
        "1.5",
        "--listener-x-m",
        "0.0",
        "--listener-y-m",
        "0.0",
        "--listener-z-m",
        "1.5",
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("generate should succeed");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    let spatial = &meta["descriptor"]["spatial"];
    assert_eq!(spatial["source_position_m"]["x"], 2.0);
    assert_eq!(spatial["source_position_m"]["y"], 9.0);
    assert_eq!(spatial["source_position_m"]["z"], 1.5);
    assert_eq!(spatial["listener_position_m"]["x"], 0.0);
    assert_eq!(spatial["listener_position_m"]["y"], 0.0);
    assert_eq!(spatial["listener_position_m"]["z"], 1.5);
}

#[test]
fn generate_rejects_partial_virtual_position_triplets() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("partial.wav");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "invalid position test",
        "--source-x-m",
        "2.0",
        "--source-y-m",
        "1.0",
        "--output",
        ir_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    let err = dispatch(cli).expect_err("partial source position should fail");
    assert!(err
        .to_string()
        .contains("source position requires all three coordinates"));
}

#[test]
fn generate_prompt_channel_hint_applies_when_no_channels_override() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("hint.wav");
    let meta_path = dir.path().join("hint.meta.json");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "massive bunker in 7.2.4",
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("parse");
    dispatch(cli).expect("generate should succeed");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["channel_format"], "atmos_7_2_4");
    assert_eq!(meta["analysis"]["channels"], 13);
}

#[test]
fn generate_rejects_layout_json_with_explicit_non_custom_channels() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("bad_layout_override.wav");
    let layout_path = dir.path().join("layout.json");

    let layout = r#"{
  "schema_version": "latent-ir.layout.v1",
  "layout_name": "bad_override_layout",
  "spatial_encoding": "discrete",
  "channels": [
    {"label":"L","azimuth_deg":30,"elevation_deg":0},
    {"label":"R","azimuth_deg":-30,"elevation_deg":0}
  ]
}"#;
    std::fs::write(&layout_path, layout).expect("write layout");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "layout override check",
        "--channels",
        "stereo",
        "--layout-json",
        layout_path.to_str().expect("utf8"),
        "--output",
        ir_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    let err = dispatch(cli).expect_err("layout-json with non-custom channels should fail");
    assert!(err
        .to_string()
        .contains("--layout-json is only valid with --channels custom"));
}

#[test]
fn generate_auto_extends_duration_for_long_t60() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("long_tail.wav");
    let meta_path = dir.path().join("long_tail.meta.json");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "huge concrete space",
        "--duration",
        "0.8",
        "--t60",
        "10.0",
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("generate should succeed");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    let duration = meta["descriptor"]["time"]["duration"]
        .as_f64()
        .unwrap_or(0.0) as f32;
    assert!(duration > 5.0);
    assert!(meta["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap_or("").contains("auto-extended")));
}

#[test]
fn generate_tail_truncation_flag_preserves_requested_short_duration() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("short_tail.wav");
    let meta_path = dir.path().join("short_tail.meta.json");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "huge concrete space",
        "--duration",
        "0.8",
        "--t60",
        "10.0",
        "--allow-tail-truncation",
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("generate should succeed");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    let duration = meta["descriptor"]["time"]["duration"]
        .as_f64()
        .unwrap_or(0.0) as f32;
    assert!(duration < 1.1);
    assert!(meta["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap_or("").contains("tail may truncate")));
}

#[test]
fn generate_tail_fade_forces_zero_file_end_and_replay_records_flag() {
    let dir = tempdir().expect("tempdir");
    let no_fade_path = dir.path().join("no_fade.wav");
    let fade_path = dir.path().join("fade.wav");
    let fade_meta_path = dir.path().join("fade.meta.json");

    let common = [
        "latent-ir",
        "generate",
        "--prompt",
        "huge concrete space",
        "--duration",
        "0.35",
        "--t60",
        "10.0",
        "--allow-tail-truncation",
        "--seed",
        "12345",
    ];

    let no_fade = Cli::try_parse_from(
        common
            .iter()
            .copied()
            .chain(["--output", no_fade_path.to_str().expect("utf8")].into_iter()),
    )
    .expect("parse");
    dispatch(no_fade).expect("generate no-fade should succeed");

    let fade = Cli::try_parse_from(
        common.iter().copied().chain(
            [
                "--tail-fade-ms",
                "40",
                "--output",
                fade_path.to_str().expect("utf8"),
                "--metadata-out",
                fade_meta_path.to_str().expect("utf8"),
            ]
            .into_iter(),
        ),
    )
    .expect("parse");
    dispatch(fade).expect("generate fade should succeed");

    let no_fade = util::audio::read_wav_f32(&no_fade_path).expect("read no-fade");
    let fade = util::audio::read_wav_f32(&fade_path).expect("read fade");
    let a = &no_fade.channels[0];
    let b = &fade.channels[0];
    assert_eq!(a.len(), b.len());

    let window = 128usize.min(a.len());
    let no_fade_tail = a[a.len() - window..].iter().map(|x| x.abs()).sum::<f32>() / window as f32;
    let fade_tail = b[b.len() - window..].iter().map(|x| x.abs()).sum::<f32>() / window as f32;
    assert!(fade_tail < no_fade_tail * 0.3);
    assert!(b.last().copied().unwrap_or(1.0).abs() <= 1e-7);

    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(&fade_meta_path).expect("meta")).unwrap();
    let replay = meta["replay_command"].as_str().unwrap_or("");
    assert!(replay.contains("--tail-fade-ms 40"));
}

#[test]
fn render_auto_resamples_ir_when_sample_rates_mismatch() {
    let dir = tempdir().expect("tempdir");
    let input_path = dir.path().join("input_48k.wav");
    let ir_path = dir.path().join("ir_44k.wav");
    let out_path = dir.path().join("rendered.wav");

    util::audio::write_wav_f32(
        &input_path,
        48_000,
        &[vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    .unwrap();
    util::audio::write_wav_f32(&ir_path, 44_100, &[vec![1.0, 0.2, 0.1, 0.0]]).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "render",
        input_path.to_str().expect("utf8"),
        "--ir",
        ir_path.to_str().expect("utf8"),
        "--auto-resample",
        "--resample-mode",
        "linear",
        "--output",
        out_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("render should succeed");
    let out = util::audio::read_wav_f32(&out_path).unwrap();
    assert_eq!(out.sample_rate, 48_000);
    assert_eq!(out.channels.len(), 1);
}

#[test]
fn morph_auto_resamples_second_ir_when_sample_rates_mismatch() {
    let dir = tempdir().expect("tempdir");
    let a_path = dir.path().join("a_48k.wav");
    let b_path = dir.path().join("b_44k.wav");
    let out_path = dir.path().join("morph.wav");

    util::audio::write_wav_f32(&a_path, 48_000, &[vec![1.0, 0.5, 0.25, 0.0]]).unwrap();
    util::audio::write_wav_f32(&b_path, 44_100, &[vec![0.0, 1.0, 0.0, 0.0]]).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "morph",
        a_path.to_str().expect("utf8"),
        b_path.to_str().expect("utf8"),
        "--alpha",
        "0.5",
        "--auto-resample",
        "--resample-mode",
        "cubic",
        "--output",
        out_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("morph should succeed");
    let out = util::audio::read_wav_f32(&out_path).unwrap();
    assert_eq!(out.sample_rate, 48_000);
    assert_eq!(out.channels.len(), 1);
}

#[test]
fn render_mismatched_sample_rates_suggest_auto_resample() {
    let dir = tempdir().expect("tempdir");
    let input_path = dir.path().join("input_48k.wav");
    let ir_path = dir.path().join("ir_44k.wav");
    let out_path = dir.path().join("rendered.wav");

    util::audio::write_wav_f32(&input_path, 48_000, &[vec![1.0, 0.0, 0.0, 0.0]]).unwrap();
    util::audio::write_wav_f32(&ir_path, 44_100, &[vec![1.0, 0.0, 0.0, 0.0]]).unwrap();

    let cli = Cli::try_parse_from([
        "latent-ir",
        "render",
        input_path.to_str().expect("utf8"),
        "--ir",
        ir_path.to_str().expect("utf8"),
        "--output",
        out_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    let err = dispatch(cli).expect_err("render should fail without auto-resample");
    assert!(err.to_string().contains("--auto-resample"));
}

#[test]
fn generate_rejects_out_of_range_sample_rate() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("bad_sr.wav");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "test",
        "--sample-rate",
        "2000",
        "--output",
        ir_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    let err = dispatch(cli).expect_err("out of range sample rate should fail");
    assert!(err.to_string().contains("supported range"));
}

#[test]
fn generate_quality_gate_lenient_passes_and_metadata_captures_gate() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("gate_pass.wav");
    let meta_path = dir.path().join("gate_pass.meta.json");

    let cli = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "short plate",
        "--duration",
        "1.0",
        "--t60",
        "0.7",
        "--quality-gate",
        "--quality-profile",
        "lenient",
        "--output",
        ir_path.to_str().expect("utf8"),
        "--metadata-out",
        meta_path.to_str().expect("utf8"),
    ])
    .expect("parse");

    dispatch(cli).expect("lenient quality gate should pass");

    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["quality_gate_profile"], "lenient");
    assert_eq!(meta["quality_gate_passed"], true);
    assert!(meta["quality_gate_failed_checks"].is_array());
}

#[test]
fn analyze_quality_gate_can_fail_and_still_write_json() {
    let dir = tempdir().expect("tempdir");
    let ir_path = dir.path().join("gate_fail.wav");
    let analysis_path = dir.path().join("gate_fail.analysis.json");

    let gen = Cli::try_parse_from([
        "latent-ir",
        "generate",
        "--prompt",
        "huge tunnel",
        "--duration",
        "0.25",
        "--t60",
        "8.0",
        "--allow-tail-truncation",
        "--output",
        ir_path.to_str().expect("utf8"),
    ])
    .expect("parse");
    dispatch(gen).expect("generate should succeed");

    let analyze = Cli::try_parse_from([
        "latent-ir",
        "analyze",
        ir_path.to_str().expect("utf8"),
        "--json",
        "--output",
        analysis_path.to_str().expect("utf8"),
        "--quality-gate",
        "--quality-profile",
        "strict",
    ])
    .expect("parse");

    let err = dispatch(analyze).expect_err("strict quality gate should fail for truncated IR");
    assert!(err.to_string().contains("quality gate failed"));
    assert!(analysis_path.exists());
}
