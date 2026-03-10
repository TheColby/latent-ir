use latent_ir::core::descriptors::DescriptorSet;
use latent_ir::core::generator::{
    generate_with_macro_trajectory, IrGenerator, ProceduralIrGenerator,
};
use latent_ir::core::perceptual::{MacroControls, MacroTrajectory};

#[test]
fn macro_controls_modify_descriptor() {
    let mut d = DescriptorSet::default();
    let t60_before = d.time.t60;
    let c = MacroControls {
        size: 0.8,
        distance: 0.6,
        material: 0.4,
        clarity: -0.3,
    };
    c.apply_to(&mut d);
    assert!(d.time.t60 > t60_before);
}

#[test]
fn trajectory_interpolates_controls() {
    let mut traj = MacroTrajectory {
        schema_version: "latent-ir.macro-trajectory.v1".to_string(),
        keyframes: vec![
            latent_ir::core::perceptual::MacroKeyframe {
                t: 0.0,
                controls: MacroControls {
                    size: -1.0,
                    distance: 0.0,
                    material: 0.0,
                    clarity: 0.0,
                },
            },
            latent_ir::core::perceptual::MacroKeyframe {
                t: 1.0,
                controls: MacroControls {
                    size: 1.0,
                    distance: 0.0,
                    material: 0.0,
                    clarity: 0.0,
                },
            },
        ],
    };
    traj.normalize().unwrap();
    let m = traj.sample(0.5);
    assert!(m.size.abs() < 1e-6);
}

#[test]
fn generator_supports_macro_trajectory() {
    let generator = ProceduralIrGenerator::new(48_000);
    let base = DescriptorSet::default();
    let mut traj = MacroTrajectory {
        schema_version: "latent-ir.macro-trajectory.v1".to_string(),
        keyframes: vec![
            latent_ir::core::perceptual::MacroKeyframe {
                t: 0.0,
                controls: MacroControls {
                    size: -0.5,
                    distance: 0.2,
                    material: -0.2,
                    clarity: 0.1,
                },
            },
            latent_ir::core::perceptual::MacroKeyframe {
                t: 1.0,
                controls: MacroControls {
                    size: 0.8,
                    distance: 0.4,
                    material: 0.6,
                    clarity: -0.3,
                },
            },
        ],
    };
    traj.normalize().unwrap();

    let out = generate_with_macro_trajectory(&generator, &base, &traj, 99).unwrap();
    assert!(!out.channels.is_empty());
    assert!(!out.channels[0].is_empty());

    let ref_static = generator.generate(&base, 99).unwrap();
    assert_ne!(out.channels[0][500], ref_static.channels[0][500]);
}
