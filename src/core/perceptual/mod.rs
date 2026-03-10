use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::core::descriptors::DescriptorSet;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MacroControls {
    pub size: f32,
    pub distance: f32,
    pub material: f32,
    pub clarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroKeyframe {
    pub t: f32,
    #[serde(default)]
    pub controls: MacroControls,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroTrajectory {
    pub schema_version: String,
    pub keyframes: Vec<MacroKeyframe>,
}

impl MacroControls {
    pub fn clamp(&mut self) {
        self.size = self.size.clamp(-1.0, 1.0);
        self.distance = self.distance.clamp(-1.0, 1.0);
        self.material = self.material.clamp(-1.0, 1.0);
        self.clarity = self.clarity.clamp(-1.0, 1.0);
    }

    pub fn apply_to(&self, d: &mut DescriptorSet) {
        let mut c = self.clone();
        c.clamp();

        d.time.t60 += c.size * 4.0;
        d.time.duration += c.size * 3.0;
        d.spatial.width += c.size * 0.2;
        d.structural.diffusion += c.size * 0.12;

        d.time.predelay_ms += c.distance * 40.0;
        d.time.attack_gap_ms += c.distance * 8.0;
        d.structural.early_density -= c.distance * 0.15;
        d.structural.late_density += c.distance * 0.1;

        d.spectral.brightness += c.material * 0.25;
        d.spectral.hf_damping -= c.material * 0.2;
        d.spectral.spectral_tilt += c.material * 0.2;

        d.structural.early_density += c.clarity * 0.15;
        d.structural.grain -= c.clarity * 0.12;
        d.structural.tail_noise -= c.clarity * 0.08;
        d.time.edt -= c.clarity * 0.4;

        d.clamp();
    }
}

impl MacroTrajectory {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).with_context(|| {
            format!(
                "failed to read macro trajectory {}",
                path.as_ref().display()
            )
        })?;
        let mut traj: Self =
            serde_json::from_str(&text).with_context(|| "failed to parse macro trajectory JSON")?;
        traj.normalize()?;
        Ok(traj)
    }

    pub fn normalize(&mut self) -> Result<()> {
        if self.keyframes.is_empty() {
            return Err(anyhow!("trajectory requires at least one keyframe"));
        }
        self.keyframes.sort_by(|a, b| a.t.total_cmp(&b.t));
        for k in &mut self.keyframes {
            k.t = k.t.clamp(0.0, 1.0);
            k.controls.clamp();
        }
        Ok(())
    }

    pub fn sample(&self, t: f32) -> MacroControls {
        let t = t.clamp(0.0, 1.0);
        if self.keyframes.len() == 1 {
            return self.keyframes[0].controls.clone();
        }

        let mut left = &self.keyframes[0];
        let mut right = &self.keyframes[self.keyframes.len() - 1];

        for w in self.keyframes.windows(2) {
            if t >= w[0].t && t <= w[1].t {
                left = &w[0];
                right = &w[1];
                break;
            }
        }

        if (right.t - left.t).abs() < 1e-6 {
            return left.controls.clone();
        }
        let a = (t - left.t) / (right.t - left.t);
        lerp_controls(&left.controls, &right.controls, a)
    }

    pub fn apply_static_average(&self, d: &mut DescriptorSet) {
        let c = self.sample(0.5);
        c.apply_to(d);
    }
}

fn lerp_controls(a: &MacroControls, b: &MacroControls, t: f32) -> MacroControls {
    let lerp = |x: f32, y: f32| x * (1.0 - t) + y * t;
    MacroControls {
        size: lerp(a.size, b.size),
        distance: lerp(a.distance, b.distance),
        material: lerp(a.material, b.material),
        clarity: lerp(a.clarity, b.clarity),
    }
}
