use approx::assert_relative_eq;
use latent_ir::core::analysis::IrAnalyzer;
use latent_ir::core::descriptors::{ChannelFormat, DescriptorSet};
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
