use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::core::descriptors::{CartesianPosition, ChannelSpec, DescriptorSet, SpatialEncoding};
use crate::core::perceptual::MacroTrajectory;

const SPEED_OF_SOUND_MPS: f32 = 343.0;

#[derive(Debug, Clone)]
pub struct GeneratedIr {
    pub channels: Vec<Vec<f32>>,
}

pub trait IrGenerator {
    fn generate(&self, descriptors: &DescriptorSet, seed: u64) -> Result<GeneratedIr>;
}

#[derive(Debug, Clone)]
pub struct ProceduralIrGenerator {
    sample_rate: u32,
}

impl ProceduralIrGenerator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
}

impl IrGenerator for ProceduralIrGenerator {
    fn generate(&self, descriptors: &DescriptorSet, seed: u64) -> Result<GeneratedIr> {
        let mono = synthesize_mono_core(descriptors, seed, self.sample_rate);
        let mut out = project_channels(
            &mono,
            descriptors,
            seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            self.sample_rate,
        );
        normalize(&mut out);
        Ok(GeneratedIr { channels: out })
    }
}

fn synthesize_mono_core(descriptors: &DescriptorSet, seed: u64, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n = (descriptors.time.duration * sr).round().max(1.0) as usize;
    let mut out = vec![0.0f32; n];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let predelay_samples = (descriptors.time.predelay_ms * 0.001 * sr) as usize;
    if predelay_samples < n {
        out[predelay_samples] = 1.0;
    }

    let early_count = (descriptors.structural.early_density * 120.0).round() as usize + 8;
    for i in 0..early_count {
        let refl = predelay_samples
            + (rng.gen_range(0.0..0.09) * sr * (0.5 + descriptors.structural.diffusion)) as usize;
        if refl >= n {
            continue;
        }

        let gain = 0.8 * (1.0 - (i as f32 / early_count as f32)).powf(1.2);
        let polarity = if rng.gen_bool(0.45) { -1.0 } else { 1.0 };
        out[refl] += gain * polarity;

        let spread = (descriptors.spatial.width * 18.0) as usize;
        let refl_b = (refl + 1 + spread + (i * 5 % 17)).min(n - 1);
        out[refl_b] += gain * 0.33;
    }

    let base_t60 = descriptors.time.t60.max(0.05);
    let low_tau = (base_t60 * descriptors.spectral.band_decay_low) / 6.91;
    let mid_tau = (base_t60 * descriptors.spectral.band_decay_mid) / 6.91;
    let high_tau = (base_t60 * descriptors.spectral.band_decay_high) / 6.91;

    for (i, sample_out) in out.iter_mut().enumerate() {
        let t = i as f32 / sr;
        let env_low = (-t / low_tau.max(0.01)).exp();
        let env_mid = (-t / mid_tau.max(0.01)).exp();
        let env_high = (-t / high_tau.max(0.01)).exp();

        let hf_mix =
            descriptors.spectral.brightness * (1.0 - descriptors.spectral.hf_damping * 0.8);
        let lf_mix = descriptors.spectral.lf_bloom;
        let mid_mix = 1.0 - (0.45 * hf_mix + 0.35 * lf_mix).clamp(0.0, 0.9);

        let density = descriptors.structural.late_density.max(0.01);
        let burst = if rng.gen_bool((0.08 + 0.72 * density) as f64) {
            rng.gen_range(-1.0..1.0)
        } else {
            0.0
        };
        let grain = descriptors.structural.grain;
        let hiss = rng.gen_range(-1.0..1.0) * descriptors.structural.tail_noise * 0.12;
        let sample = burst * (lf_mix * env_low + mid_mix * env_mid + hf_mix * env_high)
            + hiss * env_high * (0.5 + grain);

        *sample_out += sample;
    }

    out
}

fn project_channels(
    mono: &[f32],
    descriptors: &DescriptorSet,
    seed: u64,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let specs = descriptors.spatial.resolved_channel_specs();
    if specs.is_empty() {
        return vec![mono.to_vec()];
    }

    if descriptors.spatial.resolved_spatial_encoding() == SpatialEncoding::AmbisonicFoaAmbix
        && specs.len() == 4
    {
        project_foa_ambix(mono, descriptors, seed, sample_rate)
    } else {
        project_discrete_layout(mono, descriptors, &specs, seed, sample_rate)
    }
}

fn project_discrete_layout(
    mono: &[f32],
    descriptors: &DescriptorSet,
    specs: &[ChannelSpec],
    seed: u64,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let n = mono.len();
    let sr = sample_rate as f32;
    let low = lowpass_onepole(mono, 0.015);
    let mut out = vec![vec![0.0f32; n]; specs.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let listener = descriptors
        .spatial
        .listener_position_m
        .unwrap_or(CartesianPosition {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
    let src = source_direction(
        descriptors.spatial.asymmetry,
        descriptors.structural.diffusion,
        descriptors.spatial.source_position_m,
        listener,
    );
    let width = descriptors.spatial.width;
    let decor = descriptors.spatial.decorrelation;
    let t60 = descriptors.time.t60.max(0.1);
    let room = image_source_room_model(descriptors);
    let min_position_distance_m = specs
        .iter()
        .filter_map(|spec| channel_distance_to_listener_m(spec, listener))
        .reduce(f32::min);

    for (ch_idx, spec) in specs.iter().enumerate() {
        let dir = channel_direction(spec);
        let alignment = dot3(src, dir).clamp(-1.0, 1.0);
        let front_focus = dir[1].max(0.0);
        let spread_gain =
            ((alignment + 1.0) * 0.5 * width + front_focus * (1.0 - width)).clamp(0.0, 1.0);
        let channel_gain = if spec.is_lfe {
            0.5
        } else {
            0.3 + 0.7 * spread_gain
        };
        let geom = geometry_model(
            spec,
            min_position_distance_m,
            listener,
            sr,
            descriptors.spectral.hf_damping,
        );

        let delay_ms =
            decor * (0.35 + ch_idx as f32 * 0.55) + (1.0 - spread_gain) * 0.9 + geom.delay_ms;
        let delay_a = (delay_ms * 0.001 * sr).round() as usize;
        let delay_b = delay_a + 1 + ((ch_idx * 7) % 13);
        let blend = (0.82 - 0.5 * decor).clamp(0.2, 0.95);
        let early_paths = image_source_paths(
            spec,
            descriptors.spatial.source_position_m,
            listener,
            room,
            sr,
            ch_idx,
        );
        let mut air_state = 0.0f32;

        for i in 0..n {
            let t = i as f32 / sr;
            let env = (-t / (0.35 * t60 + 0.05)).exp();
            let noise = rng.gen_range(-1.0..1.0) * descriptors.structural.tail_noise * 0.02 * env;

            let mut sample = if spec.is_lfe {
                delayed_sample(&low, i, delay_a) * channel_gain + noise * 0.15
            } else {
                let base = delayed_sample(mono, i, delay_a);
                let smear = delayed_sample(mono, i, delay_b);
                let mut reflected = 0.0f32;
                for path in &early_paths {
                    reflected += delayed_sample(mono, i, delay_a + path.delay_samples) * path.gain;
                }
                ((base * blend + smear * (1.0 - blend)) + reflected) * channel_gain + noise
            };

            sample *= geom.distance_gain;

            if !spec.is_lfe && geom.air_blend > 1e-6 {
                air_state += geom.air_alpha * (sample - air_state);
                sample = sample * (1.0 - geom.air_blend) + air_state * geom.air_blend;
            }

            out[ch_idx][i] = sample;
        }
    }

    out
}

fn project_foa_ambix(
    mono: &[f32],
    descriptors: &DescriptorSet,
    seed: u64,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let n = mono.len();
    let sr = sample_rate as f32;

    let mut w = vec![0.0f32; n];
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut z = vec![0.0f32; n];

    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xD1B5_4A32_C19F_2A67);
    let listener = descriptors
        .spatial
        .listener_position_m
        .unwrap_or(CartesianPosition {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
    let src = source_direction(
        descriptors.spatial.asymmetry,
        descriptors.structural.diffusion,
        descriptors.spatial.source_position_m,
        listener,
    );
    let width = descriptors.spatial.width;
    let decor = descriptors.spatial.decorrelation;
    let t60 = descriptors.time.t60.max(0.1);

    let delay_x = (decor * 0.0015 * sr).round() as usize + 1;
    let delay_y = (decor * 0.0022 * sr).round() as usize + 2;
    let delay_z = (decor * 0.0031 * sr).round() as usize + 3;

    let dir_gain = 0.25 + 0.65 * width;
    let z_bias = 0.15 + 0.35 * (1.0 - descriptors.structural.diffusion);

    for i in 0..n {
        let t = i as f32 / sr;
        let env = (-t / (0.4 * t60 + 0.03)).exp();

        let b0 = mono[i];
        let bx = mono[i.saturating_sub(delay_x)];
        let by = mono[i.saturating_sub(delay_y)];
        let bz = mono[i.saturating_sub(delay_z)];

        let noise_scale = descriptors.structural.tail_noise * 0.015 * env;
        let nx = rng.gen_range(-1.0..1.0) * noise_scale;
        let ny = rng.gen_range(-1.0..1.0) * noise_scale;
        let nz = rng.gen_range(-1.0..1.0) * noise_scale;

        w[i] = b0 * 0.70710677;
        x[i] = dir_gain * (src[0] * bx + 0.35 * (bx - by)) + nx;
        y[i] = dir_gain * (src[1] * by + 0.35 * (by - bz)) + ny;
        z[i] = dir_gain * (z_bias * bz + 0.25 * (bx - bz)) + nz;
    }

    vec![w, x, y, z]
}

fn source_direction(
    asymmetry: f32,
    diffusion: f32,
    source_position_m: Option<CartesianPosition>,
    listener_position_m: CartesianPosition,
) -> [f32; 3] {
    if let Some(src) = source_position_m {
        let mut v = [
            src.x - listener_position_m.x,
            src.y - listener_position_m.y,
            src.z - listener_position_m.z,
        ];
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if norm > 1e-6 {
            v[0] /= norm;
            v[1] /= norm;
            v[2] /= norm;
            return v;
        }
    }

    let yaw = asymmetry.clamp(-1.0, 1.0) * std::f32::consts::FRAC_PI_4;
    let mut v = [
        yaw.sin(),
        yaw.cos(),
        0.15 + 0.35 * (1.0 - diffusion.clamp(0.0, 1.0)),
    ];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
    v
}

fn channel_direction(spec: &ChannelSpec) -> [f32; 3] {
    let az = spec.azimuth_deg.to_radians();
    let el = spec.elevation_deg.to_radians();

    let mut v = [az.sin() * el.cos(), az.cos() * el.cos(), el.sin()];
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
    v
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Debug, Clone, Copy)]
struct ChannelGeometryModel {
    delay_ms: f32,
    distance_gain: f32,
    air_blend: f32,
    air_alpha: f32,
}

#[derive(Debug, Clone, Copy)]
struct RoomModel {
    half_extents_m: [f32; 3],
    wall_reflectivity: [f32; 3],
    max_early_extra_m: f32,
}

#[derive(Debug, Clone, Copy)]
struct ImageSourcePath {
    delay_samples: usize,
    gain: f32,
}

fn image_source_room_model(descriptors: &DescriptorSet) -> RoomModel {
    let size_scale = (descriptors.time.t60 / 2.4).clamp(0.4, 6.0);
    let width = descriptors.spatial.width.clamp(0.0, 1.0);
    let diffusion = descriptors.structural.diffusion.clamp(0.0, 1.0);
    let early_density = descriptors.structural.early_density.clamp(0.0, 1.0);

    let half_x = (5.5 * size_scale * (0.75 + 0.5 * width)).clamp(2.0, 60.0);
    let half_y = (7.0 * size_scale * (0.8 + 0.4 * diffusion)).clamp(2.5, 80.0);
    let half_z = (2.7 + 1.4 * size_scale).clamp(2.2, 20.0);

    let base_reflect =
        (0.45 + 0.35 * (1.0 - descriptors.spectral.hf_damping) + 0.2 * diffusion).clamp(0.3, 0.95);
    let wall_reflectivity = [
        (base_reflect * (0.95 + 0.1 * width)).clamp(0.25, 0.98),
        (base_reflect * (1.0 + 0.12 * (1.0 - width))).clamp(0.25, 0.98),
        (base_reflect * 0.9).clamp(0.25, 0.98),
    ];

    let early_window_s = 0.02 + 0.11 * (0.35 + 0.65 * early_density) * (0.7 + 0.3 * diffusion);
    let max_early_extra_m = early_window_s * SPEED_OF_SOUND_MPS;

    RoomModel {
        half_extents_m: [half_x, half_y, half_z],
        wall_reflectivity,
        max_early_extra_m,
    }
}

fn image_source_paths(
    spec: &ChannelSpec,
    source_position_m: Option<CartesianPosition>,
    listener_position_m: CartesianPosition,
    room: RoomModel,
    sample_rate: f32,
    channel_index: usize,
) -> Vec<ImageSourcePath> {
    let (source, mic) = match (source_position_m, spec.position_m) {
        (Some(source), Some(mic)) => (source, mic),
        _ => return Vec::new(),
    };

    let direct = distance_m(source, mic).max(0.01);
    let mut out = Vec::with_capacity(6);

    for axis in 0..3 {
        let half = room.half_extents_m[axis];
        for (wall_idx, wall_sign) in [1.0f32, -1.0f32].iter().enumerate() {
            let wall_coord = component(listener_position_m, axis) + wall_sign * half;
            let mut image = source;
            set_component(&mut image, axis, 2.0 * wall_coord - component(source, axis));

            let refl = distance_m(image, mic);
            let extra = refl - direct;
            if extra <= 0.02 || extra > room.max_early_extra_m {
                continue;
            }

            let delay_samples = ((extra / SPEED_OF_SOUND_MPS) * sample_rate).round() as usize;
            if delay_samples == 0 {
                continue;
            }

            let distance_term = (direct / refl.max(direct)).clamp(0.1, 1.0).powf(0.72);
            let mut gain = 0.26 * room.wall_reflectivity[axis] * distance_term;
            if (channel_index + axis + wall_idx) % 2 == 1 {
                gain *= 0.92;
            }

            out.push(ImageSourcePath {
                delay_samples,
                gain,
            });
        }
    }

    out.sort_by_key(|p| p.delay_samples);
    if out.len() > 6 {
        out.truncate(6);
    }
    out
}

fn geometry_model(
    spec: &ChannelSpec,
    min_position_distance_m: Option<f32>,
    listener_position_m: CartesianPosition,
    sample_rate: f32,
    hf_damping: f32,
) -> ChannelGeometryModel {
    if let (Some(dist_m), Some(min_dist_m)) = (
        channel_distance_to_listener_m(spec, listener_position_m),
        min_position_distance_m,
    ) {
        let dist_m = dist_m.max(0.01);
        let relative_m = (dist_m - min_dist_m).max(0.0);
        let delay_ms = relative_m * 1000.0 / SPEED_OF_SOUND_MPS;

        // Mild inverse-distance style shaping: 1 m is neutral.
        let distance_gain = 1.0 / (1.0 + 0.08 * (dist_m - 1.0).max(0.0));

        // Higher distances introduce progressively stronger HF absorption.
        let dist_excess = (dist_m - 1.0).max(0.0);
        let air_blend =
            (dist_excess / 30.0).clamp(0.0, 0.75) * (0.35 + 0.65 * hf_damping.clamp(0.0, 1.0));
        let cutoff_hz = (18_000.0 / (1.0 + 0.12 * dist_excess)).clamp(1_500.0, 18_000.0);
        let air_alpha = (1.0
            - (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate.max(1.0)).exp())
        .clamp(0.02, 1.0);

        ChannelGeometryModel {
            delay_ms,
            distance_gain,
            air_blend,
            air_alpha,
        }
    } else {
        ChannelGeometryModel {
            delay_ms: 0.0,
            distance_gain: 1.0,
            air_blend: 0.0,
            air_alpha: 1.0,
        }
    }
}

fn channel_distance_to_listener_m(
    spec: &ChannelSpec,
    listener_position_m: CartesianPosition,
) -> Option<f32> {
    spec.position_m.map(|pos| {
        let dx = pos.x - listener_position_m.x;
        let dy = pos.y - listener_position_m.y;
        let dz = pos.z - listener_position_m.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    })
}

fn distance_m(a: CartesianPosition, b: CartesianPosition) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn component(p: CartesianPosition, axis: usize) -> f32 {
    match axis {
        0 => p.x,
        1 => p.y,
        _ => p.z,
    }
}

fn set_component(p: &mut CartesianPosition, axis: usize, v: f32) {
    match axis {
        0 => p.x = v,
        1 => p.y = v,
        _ => p.z = v,
    }
}

fn delayed_sample(signal: &[f32], i: usize, delay: usize) -> f32 {
    if i < delay {
        0.0
    } else {
        signal[i - delay]
    }
}

fn lowpass_onepole(x: &[f32], alpha: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let mut prev = 0.0;
    for (i, &v) in x.iter().enumerate() {
        prev += alpha * (v - prev);
        y[i] = prev;
    }
    y
}

fn normalize(channels: &mut [Vec<f32>]) {
    let mut peak = 0.0f32;
    for ch in channels.iter() {
        for &s in ch {
            peak = peak.max(s.abs());
        }
    }
    if peak <= 1e-9 {
        return;
    }
    let gain = 0.98 / peak;
    for ch in channels {
        for s in ch {
            *s *= gain;
        }
    }
}

pub fn generate_with_macro_trajectory(
    generator: &ProceduralIrGenerator,
    base: &DescriptorSet,
    trajectory: &MacroTrajectory,
    seed: u64,
) -> Result<GeneratedIr> {
    let segments = 8usize;
    let mut sum: Option<Vec<Vec<f32>>> = None;
    let mut weight_sum: Option<Vec<f32>> = None;

    for s in 0..segments {
        let t = (s as f32 + 0.5) / segments as f32;
        let mut d = base.clone();
        trajectory.sample(t).apply_to(&mut d);
        let part = generator.generate(&d, seed.wrapping_add((s * 7919) as u64))?;

        let n = part.channels.first().map(Vec::len).unwrap_or(0);
        let channels = part.channels.len();
        if sum.is_none() {
            sum = Some(vec![vec![0.0f32; n]; channels]);
            weight_sum = Some(vec![0.0f32; n]);
        }
        let sum_ref = sum.as_mut().expect("sum must exist");
        let w_ref = weight_sum.as_mut().expect("weights must exist");
        let target_n = w_ref.len();

        let center = (t * target_n as f32) as isize;
        let spread = (target_n as f32 / segments as f32).max(1.0);
        for i in 0..target_n {
            let dx = (i as isize - center) as f32 / spread;
            let w = (-0.5 * dx * dx).exp();
            w_ref[i] += w;
            for ch in 0..channels {
                let v = part.channels[ch].get(i).copied().unwrap_or(0.0);
                sum_ref[ch][i] += v * w;
            }
        }
    }

    let mut out = sum.unwrap_or_else(|| vec![vec![]]);
    if let Some(w_ref) = weight_sum {
        for (i, &w) in w_ref.iter().enumerate() {
            let wn = if w <= 1e-9 { 1.0 } else { w };
            for ch in &mut out {
                ch[i] /= wn;
            }
        }
    }
    normalize(&mut out);
    Ok(GeneratedIr { channels: out })
}
