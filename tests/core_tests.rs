use approx::assert_relative_eq;
use latent_ir::core::analysis::IrAnalyzer;
use latent_ir::core::descriptors::{
    CartesianPosition, ChannelFormat, ChannelSpec, CustomChannelLayout, DescriptorSet,
    SpatialEncoding,
};
use latent_ir::core::generator::{IrGenerator, ProceduralIrGenerator};
use latent_ir::core::morph::IrMorpher;
use latent_ir::core::presets;
use latent_ir::core::render::{RenderEngine, RenderOptions, Renderer};
use latent_ir::core::semantics::SemanticResolver;

#[test]
fn preset_loading_works() {
    let d = presets::resolve_preset("dark_stone_cathedral").expect("preset should resolve");
    assert!(d.time.t60 > 8.0);
    assert!(d.spectral.hf_damping > 0.7);
}

#[test]
fn semantics_adjusts_descriptors() {
    let mut d = DescriptorSet::default();
    let baseline = d.time.t60;
    SemanticResolver::default().apply_prompt("vast icy cathedral", &mut d);
    assert!(d.time.t60 > baseline);
    assert!(d.spectral.brightness > 0.5);
}

#[test]
fn semantics_extracts_rt60_from_prompt() {
    let mut d = DescriptorSet::default();
    SemanticResolver::default().apply_prompt(
        "massive concrete silo with an rt60 of around 27 seconds",
        &mut d,
    );
    assert!((d.time.t60 - 27.0).abs() < 0.5);
}

#[test]
fn semantics_parses_concrete_thickness() {
    let mut thin = DescriptorSet::default();
    SemanticResolver::default().apply_prompt("concrete bunker 0.5 ft wall", &mut thin);

    let mut thick = DescriptorSet::default();
    SemanticResolver::default().apply_prompt("concrete bunker 3 ft poured concrete", &mut thick);

    assert!(thick.spectral.hf_damping > thin.spectral.hf_damping);
    assert!(thick.structural.modal_density > thin.structural.modal_density);
}

#[test]
fn semantics_parses_predelay_and_duration_from_prompt() {
    let mut d = DescriptorSet::default();
    SemanticResolver::default().apply_prompt("long ir duration 12 seconds, predelay 65 ms", &mut d);
    assert!((d.time.duration - 12.0).abs() < 0.2);
    assert!((d.time.predelay_ms - 65.0).abs() < 1.0);
}

#[test]
fn semantics_resolves_channel_hints_from_prompt() {
    let mut d = DescriptorSet::default();
    SemanticResolver::default().apply_prompt("massive bunker in 7.2.4", &mut d);
    assert_eq!(d.spatial.channel_format, ChannelFormat::Atmos7_2_4);

    let mut e = DescriptorSet::default();
    SemanticResolver::default().apply_prompt("stereo intimate wood chapel", &mut e);
    assert_eq!(e.spatial.channel_format, ChannelFormat::Stereo);
}

#[test]
fn deterministic_generation_by_seed() {
    let d = DescriptorSet::default();
    let g = ProceduralIrGenerator::new(48_000);
    let a = g.generate(&d, 123).unwrap();
    let b = g.generate(&d, 123).unwrap();
    assert_eq!(a.channels.len(), b.channels.len());
    assert_eq!(a.channels[0].len(), b.channels[0].len());
    assert_relative_eq!(a.channels[0][100], b.channels[0][100], epsilon = 1e-7);
}

#[test]
fn foa_generation_emits_four_channels() {
    let mut d = DescriptorSet::default();
    d.spatial.channel_format = ChannelFormat::FoaAmbix;
    d.time.duration = 0.25;
    let g = ProceduralIrGenerator::new(48_000);
    let out = g.generate(&d, 2026).unwrap();
    assert_eq!(out.channels.len(), 4);
    assert!(out.channels.iter().all(|ch| !ch.is_empty()));
}

#[test]
fn atmos_7_2_4_generation_emits_thirteen_channels() {
    let mut d = DescriptorSet::default();
    d.spatial.channel_format = ChannelFormat::Atmos7_2_4;
    d.time.duration = 0.2;
    let g = ProceduralIrGenerator::new(48_000);
    let out = g.generate(&d, 2027).unwrap();
    assert_eq!(out.channels.len(), 13);
    assert!(out.channels.iter().all(|ch| !ch.is_empty()));
}

#[test]
fn custom_geometry_position_affects_delay_gain_and_hf_balance() {
    let mut d = DescriptorSet::default();
    d.time.duration = 0.5;
    d.time.predelay_ms = 0.0;
    d.time.t60 = 0.8;
    d.structural.tail_noise = 0.0;
    d.structural.grain = 0.0;
    d.structural.early_density = 0.0;
    d.structural.diffusion = 0.35;
    d.spatial.width = 0.5;
    d.spatial.decorrelation = 0.1;

    d.spatial.set_custom_layout(CustomChannelLayout {
        layout_name: "geom_test".to_string(),
        spatial_encoding: SpatialEncoding::Discrete,
        channels: vec![
            ChannelSpec {
                label: "near".to_string(),
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                is_lfe: false,
                position_m: Some(CartesianPosition {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                }),
            },
            ChannelSpec {
                label: "far".to_string(),
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                is_lfe: false,
                position_m: Some(CartesianPosition {
                    x: 0.0,
                    y: 20.0,
                    z: 0.0,
                }),
            },
        ],
    });

    let g = ProceduralIrGenerator::new(48_000);
    let out = g.generate(&d, 4242).unwrap();
    assert_eq!(out.channels.len(), 2);

    let near = &out.channels[0];
    let far = &out.channels[1];

    let near_centroid = energy_centroid_samples(near);
    let far_centroid = energy_centroid_samples(far);
    assert!(far_centroid > near_centroid + 1000.0);

    let near_rms = rms(near);
    let far_rms = rms(far);
    assert!(far_rms < near_rms * 0.85);

    let near_hf = hf_proxy(near);
    let far_hf = hf_proxy(far);
    assert!(far_hf < near_hf * 0.95);
}

#[test]
fn virtual_source_and_listener_positions_steer_projection() {
    let mut base = DescriptorSet::default();
    base.time.duration = 0.4;
    base.structural.tail_noise = 0.0;
    base.structural.grain = 0.0;
    base.structural.early_density = 0.0;
    base.structural.diffusion = 0.35;
    base.spatial.width = 0.6;
    base.spatial.decorrelation = 0.15;
    base.spatial.set_custom_layout(CustomChannelLayout {
        layout_name: "quad".to_string(),
        spatial_encoding: SpatialEncoding::Discrete,
        channels: vec![
            ChannelSpec {
                label: "F".to_string(),
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                is_lfe: false,
                position_m: Some(CartesianPosition {
                    x: 0.0,
                    y: 10.0,
                    z: 0.0,
                }),
            },
            ChannelSpec {
                label: "R".to_string(),
                azimuth_deg: 180.0,
                elevation_deg: 0.0,
                is_lfe: false,
                position_m: Some(CartesianPosition {
                    x: 0.0,
                    y: -10.0,
                    z: 0.0,
                }),
            },
            ChannelSpec {
                label: "L".to_string(),
                azimuth_deg: -90.0,
                elevation_deg: 0.0,
                is_lfe: false,
                position_m: Some(CartesianPosition {
                    x: -10.0,
                    y: 0.0,
                    z: 0.0,
                }),
            },
            ChannelSpec {
                label: "Rt".to_string(),
                azimuth_deg: 90.0,
                elevation_deg: 0.0,
                is_lfe: false,
                position_m: Some(CartesianPosition {
                    x: 10.0,
                    y: 0.0,
                    z: 0.0,
                }),
            },
        ],
    });

    let g = ProceduralIrGenerator::new(48_000);

    let mut src_front = base.clone();
    src_front.spatial.source_position_m = Some(CartesianPosition {
        x: 0.0,
        y: 25.0,
        z: 0.0,
    });
    let ir_front = g.generate(&src_front, 777).unwrap();

    let mut src_rear = base.clone();
    src_rear.spatial.source_position_m = Some(CartesianPosition {
        x: 0.0,
        y: -25.0,
        z: 0.0,
    });
    let ir_rear = g.generate(&src_rear, 777).unwrap();

    let front_f = rms(&ir_front.channels[0]);
    let rear_f = rms(&ir_front.channels[1]);
    let front_r = rms(&ir_rear.channels[0]);
    let rear_r = rms(&ir_rear.channels[1]);
    assert!(front_f > rear_f);
    assert!(rear_r > front_r);

    let mut listener_shifted = base.clone();
    listener_shifted.spatial.listener_position_m = Some(CartesianPosition {
        x: 0.0,
        y: 8.0,
        z: 0.0,
    });
    let ir_origin = g.generate(&base, 777).unwrap();
    let ir_shifted = g.generate(&listener_shifted, 777).unwrap();

    let rear_origin_centroid = energy_centroid_samples(&ir_origin.channels[1]);
    let rear_shifted_centroid = energy_centroid_samples(&ir_shifted.channels[1]);
    assert!(rear_shifted_centroid > rear_origin_centroid + 800.0);
}

#[test]
fn image_source_lite_adds_early_reflection_energy() {
    let mut d = DescriptorSet::default();
    d.time.duration = 0.7;
    d.time.predelay_ms = 0.0;
    d.time.t60 = 1.6;
    d.structural.early_density = 0.0;
    d.structural.diffusion = 0.45;
    d.structural.tail_noise = 0.0;
    d.structural.grain = 0.0;
    d.spatial.width = 0.6;
    d.spatial.decorrelation = 0.12;
    d.spatial.listener_position_m = Some(CartesianPosition {
        x: 0.0,
        y: 0.0,
        z: 1.5,
    });
    d.spatial.source_position_m = Some(CartesianPosition {
        x: 0.0,
        y: 2.5,
        z: 1.5,
    });
    d.spatial.set_custom_layout(CustomChannelLayout {
        layout_name: "image_source_probe".to_string(),
        spatial_encoding: SpatialEncoding::Discrete,
        channels: vec![ChannelSpec {
            label: "mic".to_string(),
            azimuth_deg: 0.0,
            elevation_deg: 0.0,
            is_lfe: false,
            position_m: Some(CartesianPosition {
                x: 0.0,
                y: 1.0,
                z: 1.5,
            }),
        }],
    });

    let mut no_image = d.clone();
    no_image.spatial.source_position_m = None;

    let g = ProceduralIrGenerator::new(48_000);
    let with_image = g.generate(&d, 2222).unwrap();
    let without_image = g.generate(&no_image, 2222).unwrap();

    let with_energy = early_window_energy(&with_image.channels[0], 48_000, 0.015, 0.12);
    let without_energy = early_window_energy(&without_image.channels[0], 48_000, 0.015, 0.12);
    let delta_energy = early_window_delta_energy(
        &with_image.channels[0],
        &without_image.channels[0],
        48_000,
        0.015,
        0.12,
    );

    assert!(with_energy > 0.0);
    assert!(without_energy > 0.0);
    assert!(delta_energy > without_energy * 0.01);
}

#[test]
fn analysis_extracts_basic_metrics() {
    let sr = 48_000;
    let len = sr as usize;
    let mut ir = vec![0.0f32; len];
    ir[240] = 1.0;
    for i in 241..len {
        let t = (i - 240) as f32 / sr as f32;
        ir[i] += (-(t / 0.8)).exp() * 0.1;
    }
    let report = IrAnalyzer::default().analyze(&[ir], sr);
    assert!(report.duration_s > 0.9);
    assert!(report.peak > 0.95);
    assert!(report.predelay_ms_est > 3.0);
    assert!(report.decay_db_span > 10.0);
    assert!(report.t60_confidence.unwrap_or(0.0) >= 0.0);
    assert!(report.t60_confidence.unwrap_or(0.0) <= 1.0);
}

#[test]
fn analysis_reports_arrival_spread_itd_and_iacc_metrics() {
    let sr = 48_000u32;
    let n = sr as usize;
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];

    a[200] = 1.0;
    b[224] = 1.0; // ~0.5 ms delayed vs channel A
    for i in 225..n {
        let ta = (i.saturating_sub(200)) as f32 / sr as f32;
        let tb = (i.saturating_sub(224)) as f32 / sr as f32;
        a[i] += (-(ta / 0.65)).exp() * 0.08;
        b[i] += (-(tb / 0.65)).exp() * 0.08;
    }

    let report = IrAnalyzer::default().analyze(&[a, b], sr);
    assert!(report.arrival_spread_ms.unwrap_or(0.0) > 0.05);
    assert!(report.arrival_max_ms.unwrap_or(0.0) > report.arrival_min_ms.unwrap_or(0.0));
    assert!((report.itd_01_ms.unwrap_or(0.0).abs() - 0.5).abs() < 0.25);
    assert!(report.iacc_early_01.unwrap_or(0.0) > 0.2);
    assert!(report.inter_channel_itd_mean_abs_ms.unwrap_or(0.0) > 0.1);
    assert!(report.inter_channel_iacc_early_mean.unwrap_or(0.0) > 0.2);
}

#[test]
fn morph_endpoints_hold() {
    let a = vec![vec![1.0f32, 0.0, 0.0]];
    let b = vec![vec![0.0f32, 1.0, 0.0]];
    let morpher = IrMorpher;

    let m0 = morpher.morph(&a, &b, 0.0);
    let m1 = morpher.morph(&a, &b, 1.0);
    assert_relative_eq!(m0[0][0], 0.98, epsilon = 1e-6);
    assert_relative_eq!(m1[0][1], 0.98, epsilon = 1e-6);
}

#[test]
fn render_pipeline_identity_for_delta_ir() {
    let input = vec![vec![0.2f32, -0.3, 0.5, 0.1]];
    let ir = vec![vec![1.0f32]];
    let out = Renderer.render_convolution(&input, &ir, 1.0);
    assert_eq!(out[0].len(), input[0].len());
    assert_relative_eq!(out[0][2], input[0][2], epsilon = 1e-6);
}

#[test]
fn render_fft_partitioned_matches_direct() {
    let input = vec![{
        let mut v = vec![0.0f32; 2048];
        for (i, s) in v.iter_mut().enumerate() {
            *s = ((i as f32 * 0.013).sin() * 0.3) + ((i as f32 * 0.031).cos() * 0.1);
        }
        v
    }];
    let ir = vec![{
        let mut v = vec![0.0f32; 1024];
        v[0] = 1.0;
        for (i, s) in v.iter_mut().enumerate().skip(1) {
            *s = (-(i as f32 / 480.0)).exp() * if i % 7 == 0 { -0.04 } else { 0.04 };
        }
        v
    }];

    let direct = Renderer.render_convolution_with_options(
        &input,
        &ir,
        1.0,
        RenderOptions {
            engine: RenderEngine::Direct,
            partition_size: 256,
        },
    );
    let fft = Renderer.render_convolution_with_options(
        &input,
        &ir,
        1.0,
        RenderOptions {
            engine: RenderEngine::FftPartitioned,
            partition_size: 256,
        },
    );

    assert_eq!(direct[0].len(), fft[0].len());
    for (a, b) in direct[0].iter().zip(fft[0].iter()) {
        assert_relative_eq!(a, b, epsilon = 2e-4);
    }
}

#[test]
fn render_fft_streaming_matches_direct() {
    let input = vec![{
        let mut v = vec![0.0f32; 3072];
        for (i, s) in v.iter_mut().enumerate() {
            *s = ((i as f32 * 0.011).sin() * 0.25) + ((i as f32 * 0.027).cos() * 0.14);
        }
        v
    }];
    let ir = vec![{
        let mut v = vec![0.0f32; 1536];
        v[0] = 1.0;
        for (i, s) in v.iter_mut().enumerate().skip(1) {
            *s = (-(i as f32 / 380.0)).exp() * if i % 11 == 0 { -0.03 } else { 0.03 };
        }
        v
    }];

    let direct = Renderer.render_convolution_with_options(
        &input,
        &ir,
        1.0,
        RenderOptions {
            engine: RenderEngine::Direct,
            partition_size: 256,
        },
    );

    let mut streamed = vec![Vec::<f32>::new(); 1];
    Renderer.render_convolution_streaming_blocks(&input, &ir, 1.0, 256, |block, valid| {
        for ch in 0..streamed.len() {
            streamed[ch].extend_from_slice(&block[ch][..valid]);
        }
    });
    let peak = streamed[0].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1e-9 {
        let gain = (0.98 / peak).min(1.0);
        for s in &mut streamed[0] {
            *s *= gain;
        }
    }

    assert_eq!(direct[0].len(), streamed[0].len());
    for (a, b) in direct[0].iter().zip(streamed[0].iter()) {
        assert_relative_eq!(a, b, epsilon = 2e-4);
    }
}

fn energy_centroid_samples(x: &[f32]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &s) in x.iter().enumerate() {
        let e = s * s;
        num += i as f32 * e;
        den += e;
    }
    if den <= 1e-12 {
        0.0
    } else {
        num / den
    }
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn hf_proxy(x: &[f32]) -> f32 {
    if x.len() < 2 {
        return 0.0;
    }
    x.windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            d * d
        })
        .sum::<f32>()
}

fn early_window_energy(x: &[f32], sample_rate: u32, t0_s: f32, t1_s: f32) -> f32 {
    let n0 = (t0_s * sample_rate as f32).round().max(0.0) as usize;
    let n1 = (t1_s * sample_rate as f32).round().max(0.0) as usize;
    let hi = n1.min(x.len());
    if n0 >= hi {
        return 0.0;
    }
    x[n0..hi].iter().map(|v| v * v).sum::<f32>()
}

fn early_window_delta_energy(a: &[f32], b: &[f32], sample_rate: u32, t0_s: f32, t1_s: f32) -> f32 {
    let n0 = (t0_s * sample_rate as f32).round().max(0.0) as usize;
    let n1 = (t1_s * sample_rate as f32).round().max(0.0) as usize;
    let hi = n1.min(a.len().min(b.len()));
    if n0 >= hi {
        return 0.0;
    }
    a[n0..hi]
        .iter()
        .zip(&b[n0..hi])
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
}
