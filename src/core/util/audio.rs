use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    Linear,
    Cubic,
}

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

pub fn read_wav_f32(path: impl AsRef<Path>) -> Result<AudioBuffer> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(anyhow!("wav has zero channels"));
    }

    let mut out = vec![Vec::new(); channels];
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            for (i, s) in reader.samples::<f32>().enumerate() {
                out[i % channels].push(s?);
            }
        }
        (SampleFormat::Int, 16) => {
            for (i, s) in reader.samples::<i16>().enumerate() {
                out[i % channels].push(s? as f32 / i16::MAX as f32);
            }
        }
        (SampleFormat::Int, 24) | (SampleFormat::Int, 32) => {
            for (i, s) in reader.samples::<i32>().enumerate() {
                out[i % channels].push(s? as f32 / i32::MAX as f32);
            }
        }
        _ => {
            return Err(anyhow!(
                "unsupported WAV format: {:?} {}-bit",
                spec.sample_format,
                spec.bits_per_sample
            ))
        }
    }

    Ok(AudioBuffer {
        sample_rate: spec.sample_rate,
        channels: out,
    })
}

pub fn write_wav_f32(
    path: impl AsRef<Path>,
    sample_rate: u32,
    channels: &[Vec<f32>],
) -> Result<()> {
    let channels_n = channels.len();
    if channels_n == 0 {
        return Err(anyhow!("cannot write zero channels"));
    }
    let len = channels.iter().map(Vec::len).max().unwrap_or(0);
    let spec = WavSpec {
        channels: channels_n as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for i in 0..len {
        for ch in channels {
            let s = ch.get(i).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
            writer.write_sample(s)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

pub fn resample_linear(channels: &[Vec<f32>], src_rate: u32, dst_rate: u32) -> Vec<Vec<f32>> {
    resample(channels, src_rate, dst_rate, ResampleMode::Linear)
}

pub fn resample(
    channels: &[Vec<f32>],
    src_rate: u32,
    dst_rate: u32,
    mode: ResampleMode,
) -> Vec<Vec<f32>> {
    if src_rate == dst_rate {
        return channels.to_vec();
    }
    channels
        .iter()
        .map(|ch| match mode {
            ResampleMode::Linear => resample_channel_linear(ch, src_rate, dst_rate),
            ResampleMode::Cubic => resample_channel_cubic(ch, src_rate, dst_rate),
        })
        .collect()
}

pub fn apply_tail_fade(
    channels: &mut [Vec<f32>],
    sample_rate: u32,
    tail_fade_ms: f32,
) -> Option<String> {
    if channels.is_empty() || tail_fade_ms <= 0.0 {
        return None;
    }
    let max_len = channels.iter().map(Vec::len).max().unwrap_or(0);
    if max_len <= 1 {
        for ch in channels {
            if let Some(s) = ch.last_mut() {
                *s = 0.0;
            }
        }
        return Some(
            "tail fade requested on near-empty IR; forcing final sample to zero".to_string(),
        );
    }

    let mut fade_n = (tail_fade_ms * 0.001 * sample_rate as f32).round() as usize;
    if fade_n < 2 {
        fade_n = 2;
    }
    let mut warning = None;
    if fade_n > max_len {
        fade_n = max_len;
        warning = Some(format!(
            "tail fade {:.2}ms exceeds IR length; clamped to full length",
            tail_fade_ms
        ));
    }

    for ch in channels {
        let n = ch.len();
        if n == 0 {
            continue;
        }
        let start = n.saturating_sub(fade_n);
        let denom = (n - start - 1).max(1) as f32;
        for i in start..n {
            let t = (i - start) as f32 / denom;
            let gain = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            ch[i] *= gain;
        }
        ch[n - 1] = 0.0;
    }
    warning
}

fn resample_channel_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }

    // Linear interpolation is intentionally boring here: deterministic, dependency-free, and good enough for CLI glue.
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_len];

    for (i, sample) in out.iter_mut().enumerate() {
        let src_pos = (i as f64) / ratio;
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = (src_pos - i0 as f64) as f32;
        let a = input[i0];
        let b = input[i1];
        *sample = a + (b - a) * frac;
    }

    out
}

fn resample_channel_cubic(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_len];

    for (i, sample) in out.iter_mut().enumerate() {
        let src_pos = (i as f64) / ratio;
        let x1 = src_pos.floor() as isize;
        let t = (src_pos - x1 as f64) as f32;

        let p0 = sample_at(input, x1 - 1);
        let p1 = sample_at(input, x1);
        let p2 = sample_at(input, x1 + 1);
        let p3 = sample_at(input, x1 + 2);

        *sample = catmull_rom(p0, p1, p2, p3, t);
    }

    out
}

fn sample_at(input: &[f32], idx: isize) -> f32 {
    if idx <= 0 {
        return input[0];
    }
    let i = idx as usize;
    if i >= input.len() {
        input[input.len() - 1]
    } else {
        input[i]
    }
}

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}
