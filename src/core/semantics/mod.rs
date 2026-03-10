use crate::core::descriptors::DescriptorSet;

#[derive(Debug, Default)]
pub struct SemanticResolver;

impl SemanticResolver {
    pub fn apply_prompt(&self, prompt: &str, descriptor: &mut DescriptorSet) {
        let p = prompt.to_ascii_lowercase();

        for token in p.split_whitespace() {
            match token {
                "cathedral" => {
                    descriptor.time.t60 += 5.0;
                    descriptor.time.predelay_ms += 20.0;
                    descriptor.structural.late_density += 0.15;
                    descriptor.spatial.width += 0.15;
                }
                "chapel" => {
                    descriptor.time.t60 += 0.8;
                    descriptor.spectral.hf_damping += 0.15;
                    descriptor.spectral.brightness -= 0.08;
                }
                "bunker" => {
                    descriptor.structural.modal_density += 0.2;
                    descriptor.structural.grain += 0.15;
                    descriptor.spectral.brightness += 0.1;
                }
                "cave" => {
                    descriptor.time.t60 += 3.0;
                    descriptor.spectral.lf_bloom += 0.18;
                    descriptor.structural.diffusion += 0.08;
                }
                "steel" => {
                    descriptor.spectral.brightness += 0.2;
                    descriptor.spectral.hf_damping -= 0.15;
                }
                "wood" => {
                    descriptor.spectral.hf_damping += 0.2;
                    descriptor.spectral.brightness -= 0.12;
                }
                "marble" => {
                    descriptor.spectral.brightness += 0.12;
                    descriptor.structural.diffusion += 0.1;
                }
                "ice" | "icy" => {
                    descriptor.spectral.brightness += 0.22;
                    descriptor.spectral.hf_damping -= 0.1;
                }
                "intimate" => {
                    descriptor.time.t60 *= 0.55;
                    descriptor.time.predelay_ms *= 0.5;
                    descriptor.spatial.width *= 0.7;
                }
                "infinite" => {
                    descriptor.time.t60 = descriptor.time.t60.max(20.0);
                    descriptor.time.duration = descriptor.time.duration.max(15.0);
                    descriptor.structural.late_density = 1.0;
                }
                "dark" => {
                    descriptor.spectral.brightness -= 0.2;
                    descriptor.spectral.hf_damping += 0.18;
                }
                "bright" => {
                    descriptor.spectral.brightness += 0.2;
                    descriptor.spectral.hf_damping -= 0.15;
                }
                _ => {}
            }
        }

        descriptor.clamp();
    }
}
