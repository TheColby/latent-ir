use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

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
