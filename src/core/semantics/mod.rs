use crate::core::descriptors::DescriptorSet;

#[derive(Debug, Default)]
pub struct SemanticResolver;

impl SemanticResolver {
    pub fn apply_prompt(&self, prompt: &str, descriptor: &mut DescriptorSet) {
        let p = prompt.to_ascii_lowercase();
        let tokens = tokenize_preserve_numbers(&p);

        apply_phrase_rules(&p, descriptor);
        apply_token_rules(&tokens, descriptor);
        apply_numeric_rules(&p, &tokens, descriptor);

        descriptor.clamp();
    }
}

fn apply_phrase_rules(prompt: &str, d: &mut DescriptorSet) {
    if prompt.contains("grain silo") || prompt.contains("silo") {
        d.time.t60 += 3.2;
        d.time.predelay_ms += 22.0;
        d.structural.modal_density += 0.18;
        d.structural.diffusion += 0.08;
        d.spatial.width += 0.08;
    }
    if prompt.contains("warehouse") || prompt.contains("hangar") {
        d.time.t60 += 2.2;
        d.time.predelay_ms += 18.0;
        d.structural.early_density -= 0.08;
        d.structural.late_density += 0.08;
    }
    if prompt.contains("cistern") || prompt.contains("tank") {
        d.time.t60 += 2.8;
        d.structural.modal_density += 0.16;
        d.structural.grain += 0.08;
    }
    if prompt.contains("poured concrete") || prompt.contains("concrete bunker") {
        d.spectral.brightness -= 0.10;
        d.spectral.hf_damping += 0.14;
        d.structural.modal_density += 0.12;
    }
    if prompt.contains("corrugated steel") {
        d.spectral.brightness += 0.16;
        d.spectral.hf_damping -= 0.08;
        d.structural.grain += 0.10;
    }
    if prompt.contains("rebar") {
        d.structural.modal_density += 0.10;
        d.structural.grain += 0.08;
    }
}

fn apply_token_rules(tokens: &[String], d: &mut DescriptorSet) {
    for token in tokens {
        match token.as_str() {
            "cathedral" => {
                d.time.t60 += 5.0;
                d.time.predelay_ms += 20.0;
                d.structural.late_density += 0.15;
                d.spatial.width += 0.15;
            }
            "chapel" => {
                d.time.t60 += 0.8;
                d.spectral.hf_damping += 0.15;
                d.spectral.brightness -= 0.08;
            }
            "bunker" => {
                d.structural.modal_density += 0.2;
                d.structural.grain += 0.15;
                d.spectral.brightness += 0.1;
            }
            "cave" => {
                d.time.t60 += 3.0;
                d.spectral.lf_bloom += 0.18;
                d.structural.diffusion += 0.08;
            }
            "steel" => {
                d.spectral.brightness += 0.2;
                d.spectral.hf_damping -= 0.15;
            }
            "wood" => {
                d.spectral.hf_damping += 0.2;
                d.spectral.brightness -= 0.12;
            }
            "concrete" => {
                d.spectral.brightness -= 0.12;
                d.spectral.hf_damping += 0.16;
                d.structural.modal_density += 0.08;
            }
            "marble" => {
                d.spectral.brightness += 0.12;
                d.structural.diffusion += 0.1;
            }
            "ice" | "icy" => {
                d.spectral.brightness += 0.22;
                d.spectral.hf_damping -= 0.1;
            }
            "intimate" => {
                d.time.t60 *= 0.55;
                d.time.predelay_ms *= 0.5;
                d.spatial.width *= 0.7;
            }
            "infinite" => {
                d.time.t60 = d.time.t60.max(20.0);
                d.time.duration = d.time.duration.max(15.0);
                d.structural.late_density = 1.0;
            }
            "dark" => {
                d.spectral.brightness -= 0.2;
                d.spectral.hf_damping += 0.18;
            }
            "bright" => {
                d.spectral.brightness += 0.2;
                d.spectral.hf_damping -= 0.15;
            }
            "massive" | "colossal" | "vast" | "huge" | "cavernous" => {
                d.time.t60 += 1.8;
                d.time.duration += 1.5;
                d.spatial.width += 0.08;
            }
            _ => {}
        }
    }
}

fn apply_numeric_rules(prompt: &str, tokens: &[String], d: &mut DescriptorSet) {
    if let Some(rt60) = extract_rt60(tokens) {
        d.time.t60 = rt60;
        d.time.duration = d.time.duration.max((rt60 * 1.1).min(30.0));
    }

    if prompt.contains("concrete") {
        if let Some(thickness_m) = extract_thickness_m(tokens) {
            let thick = (thickness_m / 0.3).clamp(0.0, 6.0);
            d.spectral.hf_damping += 0.03 * thick;
            d.structural.modal_density += 0.025 * thick;
            d.spectral.brightness -= 0.02 * thick;
        }
    }
}

fn tokenize_preserve_numbers(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse_f32(tok: &str) -> Option<f32> {
    tok.parse::<f32>().ok()
}

fn extract_rt60(tokens: &[String]) -> Option<f32> {
    for i in 0..tokens.len() {
        let t = tokens[i].as_str();
        if (t == "rt60" || t == "t60") && i + 1 < tokens.len() {
            if let Some(v) = parse_f32(&tokens[i + 1]) {
                return Some(v.clamp(0.1, 60.0));
            }
        }
        if let Some(v) = parse_f32(t) {
            if i + 1 < tokens.len()
                && (tokens[i + 1] == "second"
                    || tokens[i + 1] == "seconds"
                    || tokens[i + 1] == "sec")
            {
                let start = i.saturating_sub(4);
                if tokens[start..i].iter().any(|w| w == "rt60" || w == "t60") {
                    return Some(v.clamp(0.1, 60.0));
                }
            }
        }
    }
    None
}

fn extract_thickness_m(tokens: &[String]) -> Option<f32> {
    for i in 0..tokens.len().saturating_sub(1) {
        let Some(v) = parse_f32(&tokens[i]) else {
            continue;
        };
        let unit = tokens[i + 1].as_str();
        let meters = match unit {
            "ft" | "foot" | "feet" => v * 0.3048,
            "m" | "meter" | "meters" => v,
            "cm" => v * 0.01,
            "in" | "inch" | "inches" => v * 0.0254,
            _ => continue,
        };
        return Some(meters.max(0.0));
    }
    None
}
