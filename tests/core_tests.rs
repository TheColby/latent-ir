use approx::assert_relative_eq;
use latent_ir::core::analysis::IrAnalyzer;
use latent_ir::core::descriptors::DescriptorSet;
use latent_ir::core::generator::{IrGenerator, ProceduralIrGenerator};
use latent_ir::core::morph::IrMorpher;
use latent_ir::core::presets;
use latent_ir::core::render::Renderer;
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
