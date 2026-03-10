use latent_ir::core::conditioning::{
    AudioEncoder, LearnedAudioEncoder, LearnedTextEncoder, TextEncoder,
};
use latent_ir::core::descriptors::DescriptorSet;

#[test]
fn learned_text_encoder_produces_nonzero_delta() {
    let model = LearnedTextEncoder::from_json_file("examples/models/text_encoder_v1.json")
        .expect("text model should load");
    let delta = model
        .infer_delta_from_prompt("dark steel cathedral")
        .expect("text inference should succeed");

    let mut d = DescriptorSet::default();
    let before = d.time.t60;
    delta.apply_to(&mut d, 1.0);
    assert!(d.time.t60 != before);
}

#[test]
fn learned_audio_encoder_produces_nonzero_delta() {
    let model = LearnedAudioEncoder::from_json_file("examples/models/audio_encoder_v1.json")
        .expect("audio model should load");

    let sr = 48_000u32;
    let n = (0.8 * sr as f32) as usize;
    let mut ir = vec![0.0f32; n];
    ir[200] = 1.0;
    for i in 201..n {
        let t = (i - 200) as f32 / sr as f32;
        ir[i] = (-(t / 0.55)).exp() * 0.15;
    }

    let delta = model
        .infer_delta_from_audio(&[ir], sr)
        .expect("audio inference should succeed");

    let mut d = DescriptorSet::default();
    let before = d.spectral.brightness;
    delta.apply_to(&mut d, 1.0);
    assert!(d.spectral.brightness != before);
}
