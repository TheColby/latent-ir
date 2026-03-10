use rustfft::{num_complex::Complex32, FftPlanner};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub schema_version: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_s: f32,
    pub peak: f32,
    pub rms: f32,
    pub predelay_ms_est: f32,
    pub edt_s_est: Option<f32>,
    pub t20_s_est: Option<f32>,
    pub t30_s_est: Option<f32>,
    pub t60_s_est: Option<f32>,
    pub spectral_centroid_hz: f32,
    pub band_decay_low_s: Option<f32>,
    pub band_decay_mid_s: Option<f32>,
    pub band_decay_high_s: Option<f32>,
    pub early_energy_ratio: f32,
    pub late_energy_ratio: f32,
    pub stereo_correlation: Option<f32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct IrAnalyzer;

impl IrAnalyzer {
    pub fn analyze(&self, channels: &[Vec<f32>], sample_rate: u32) -> AnalysisReport {
        let mut warnings = Vec::new();
        let c = channels.len().max(1);
        let n = channels.first().map(|x| x.len()).unwrap_or(0);
        let duration_s = n as f32 / sample_rate as f32;

        let mono = downmix(channels);
        let peak = mono.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let rms = if mono.is_empty() {
            0.0
        } else {
            (mono.iter().map(|x| x * x).sum::<f32>() / mono.len() as f32).sqrt()
        };

        let predelay_ms_est = estimate_predelay_ms(&mono, sample_rate);
        let (edt, t20, t30, t60) = estimate_decay_times(&mono, sample_rate);

        if t60.is_none() {
            warnings.push("T60 estimate unavailable due to insufficient decay range".to_string());
        }
        if c > 2 {
            warnings.push(
                "stereo_correlation reports channel 0/1 pair only for multichannel material"
                    .to_string(),
            );
        }

        let spectral_centroid_hz = estimate_centroid(&mono, sample_rate);
        let (low, mid, high) = estimate_band_decays(&mono, sample_rate);
        let (early_ratio, late_ratio) = energy_ratios(&mono, sample_rate, 0.08);
        let stereo_corr = if c >= 2 {
            Some(correlation(&channels[0], &channels[1]))
        } else {
            None
        };

        AnalysisReport {
            schema_version: "latent-ir.analysis.v1".to_string(),
            sample_rate,
            channels: c,
            duration_s,
            peak,
            rms,
            predelay_ms_est,
            edt_s_est: edt,
            t20_s_est: t20,
            t30_s_est: t30,
            t60_s_est: t60,
            spectral_centroid_hz,
            band_decay_low_s: low,
            band_decay_mid_s: mid,
            band_decay_high_s: high,
            early_energy_ratio: early_ratio,
            late_energy_ratio: late_ratio,
            stereo_correlation: stereo_corr,
            warnings,
        }
    }
}

fn downmix(channels: &[Vec<f32>]) -> Vec<f32> {
    match channels {
        [] => vec![],
        [mono] => mono.clone(),
        many => {
            let n = many[0].len();
            let inv = 1.0 / many.len() as f32;
            let mut out = vec![0.0f32; n];
            for ch in many {
                for (o, &s) in out.iter_mut().zip(ch) {
                    *o += s * inv;
                }
            }
            out
        }
    }
}

fn estimate_predelay_ms(ir: &[f32], sr: u32) -> f32 {
    if ir.is_empty() {
        return 0.0;
    }
    let peak = ir.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let thresh = peak * 0.1;
    let idx = ir.iter().position(|x| x.abs() >= thresh).unwrap_or(0);
    idx as f32 * 1000.0 / sr as f32
}

fn schroeder_db(ir: &[f32]) -> Vec<f32> {
    let mut edc = vec![0.0f32; ir.len()];
    let mut acc = 0.0f32;
    for (i, &x) in ir.iter().enumerate().rev() {
        acc += x * x;
        edc[i] = acc;
    }
    let max = edc.first().copied().unwrap_or(1e-12).max(1e-12);
    edc.iter_mut().for_each(|e| {
        *e = 10.0 * ((*e / max).max(1e-12)).log10();
    });
    edc
}

fn linear_fit(xs: &[f32], ys: &[f32]) -> Option<(f32, f32)> {
    if xs.len() < 8 || xs.len() != ys.len() {
        return None;
    }
    let n = xs.len() as f32;
    let sx = xs.iter().sum::<f32>();
    let sy = ys.iter().sum::<f32>();
    let sxx = xs.iter().map(|x| x * x).sum::<f32>();
    let sxy = xs.iter().zip(ys).map(|(x, y)| x * y).sum::<f32>();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        return None;
    }
    let m = (n * sxy - sx * sy) / denom;
    let b = (sy - m * sx) / n;
    Some((m, b))
}

fn segment_fit_time(edc_db: &[f32], sr: u32, hi_db: f32, lo_db: f32) -> Option<f32> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, &db) in edc_db.iter().enumerate() {
        if db <= hi_db && db >= lo_db {
            xs.push(i as f32 / sr as f32);
            ys.push(db);
        }
    }
    let (m, _b) = linear_fit(&xs, &ys)?;
    if m >= 0.0 {
        return None;
    }
    Some((-60.0 / m).abs())
}

fn estimate_decay_times(
    ir: &[f32],
    sr: u32,
) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    let edc = schroeder_db(ir);
    let edt = segment_fit_time(&edc, sr, 0.0, -10.0).map(|v| v / 6.0);
    let t20 = segment_fit_time(&edc, sr, -5.0, -25.0).map(|v| v / 3.0);
    let t30 = segment_fit_time(&edc, sr, -5.0, -35.0).map(|v| v / 2.0);
    let t60 = t30.or(t20).or(edt.map(|x| x * 1.5));
    (edt, t20, t30, t60)
}

fn estimate_centroid(ir: &[f32], sr: u32) -> f32 {
    if ir.len() < 8 {
        return 0.0;
    }
    let n_fft = ir.len().next_power_of_two().min(16384);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
    for (i, s) in ir.iter().take(n_fft).enumerate() {
        buf[i].re = *s;
    }
    fft.process(&mut buf);

    let bin_hz = sr as f32 / n_fft as f32;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, c) in buf.iter().take(n_fft / 2).enumerate() {
        let mag = c.norm();
        let f = i as f32 * bin_hz;
        num += f * mag;
        den += mag;
    }
    if den <= 1e-9 {
        0.0
    } else {
        num / den
    }
}

fn estimate_band_decays(ir: &[f32], sr: u32) -> (Option<f32>, Option<f32>, Option<f32>) {
    let low = lowpass_onepole(ir, 0.02);
    let high = highpass_onepole(ir, 0.18);
    let mid: Vec<f32> = ir
        .iter()
        .zip(&low)
        .zip(&high)
        .map(|((&x, &l), &h)| x - l - h)
        .collect();
    let (_, _, _, l60) = estimate_decay_times(&low, sr);
    let (_, _, _, m60) = estimate_decay_times(&mid, sr);
    let (_, _, _, h60) = estimate_decay_times(&high, sr);
    (l60, m60, h60)
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

fn highpass_onepole(x: &[f32], alpha: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let mut lp = 0.0;
    for (i, &v) in x.iter().enumerate() {
        lp += alpha * (v - lp);
        y[i] = v - lp;
    }
    y
}

fn energy_ratios(ir: &[f32], sr: u32, early_s: f32) -> (f32, f32) {
    let split = (early_s * sr as f32).round() as usize;
    let split = split.min(ir.len());
    let early = ir[..split].iter().map(|x| x * x).sum::<f32>();
    let late = ir[split..].iter().map(|x| x * x).sum::<f32>();
    let total = (early + late).max(1e-12);
    (early / total, late / total)
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xa = x - ma;
        let yb = y - mb;
        num += xa * yb;
        da += xa * xa;
        db += yb * yb;
    }
    if da <= 1e-12 || db <= 1e-12 {
        0.0
    } else {
        (num / (da.sqrt() * db.sqrt())).clamp(-1.0, 1.0)
    }
}
