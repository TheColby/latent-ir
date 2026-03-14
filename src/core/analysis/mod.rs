use rustfft::{num_complex::Complex32, FftPlanner};
use serde::{Deserialize, Serialize};

use crate::core::spatial::{self, ChannelMap};

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
    pub inter_channel_correlation_matrix: Option<Vec<Vec<f32>>>,
    pub inter_channel_correlation_mean_abs: Option<f32>,
    pub inter_channel_correlation_min_abs: Option<f32>,
    pub arrival_min_ms: Option<f32>,
    pub arrival_max_ms: Option<f32>,
    pub arrival_spread_ms: Option<f32>,
    pub itd_01_ms: Option<f32>,
    pub iacc_early_01: Option<f32>,
    pub inter_channel_itd_mean_abs_ms: Option<f32>,
    pub inter_channel_itd_max_abs_ms: Option<f32>,
    pub inter_channel_iacc_early_mean: Option<f32>,
    pub inter_channel_iacc_early_min: Option<f32>,
    pub front_energy_ratio: Option<f32>,
    pub rear_energy_ratio: Option<f32>,
    pub height_energy_ratio: Option<f32>,
    pub lfe_energy_ratio: Option<f32>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct IrAnalyzer;

impl IrAnalyzer {
    pub fn analyze(&self, channels: &[Vec<f32>], sample_rate: u32) -> AnalysisReport {
        self.analyze_with_channel_map(channels, sample_rate, None)
    }

    pub fn analyze_with_channel_map(
        &self,
        channels: &[Vec<f32>],
        sample_rate: u32,
        channel_map: Option<&ChannelMap>,
    ) -> AnalysisReport {
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

        let spectral_centroid_hz = estimate_centroid(&mono, sample_rate);
        let (low, mid, high) = estimate_band_decays(&mono, sample_rate);
        let (early_ratio, late_ratio) = energy_ratios(&mono, sample_rate, 0.08);
        let stereo_corr = if c >= 2 {
            Some(correlation(&channels[0], &channels[1]))
        } else {
            None
        };

        let (corr_matrix, corr_mean_abs, corr_min_abs) = inter_channel_correlation(channels);
        let (arrival_min_ms, arrival_max_ms, arrival_spread_ms) =
            arrival_spread_metrics(channels, sample_rate);
        let (itd_01_ms, iacc_early_01) = if c >= 2 {
            if let Some((lag_ms, iacc)) =
                estimate_itd_iacc(&channels[0], &channels[1], sample_rate, 0.08, 0.001)
            {
                (Some(lag_ms), Some(iacc))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let (
            inter_channel_itd_mean_abs_ms,
            inter_channel_itd_max_abs_ms,
            inter_channel_iacc_early_mean,
            inter_channel_iacc_early_min,
            sampled_pairs,
            total_pairs,
        ) = inter_channel_itd_iacc_metrics(channels, sample_rate, 0.08, 0.001, 96);

        let (front_energy_ratio, rear_energy_ratio, height_energy_ratio, lfe_energy_ratio) =
            if let Some(map) = channel_map {
                match spatial::validate_channel_map(map, c)
                    .and_then(|_| directional_energy_ratios(channels, map))
                {
                    Ok(v) => v,
                    Err(err) => {
                        warnings.push(format!("directional energy metrics unavailable: {err}"));
                        (None, None, None, None)
                    }
                }
            } else {
                if c > 2 {
                    warnings.push(
                        "directional energy metrics unavailable: no channel map provided"
                            .to_string(),
                    );
                }
                (None, None, None, None)
            };

        if c > 2 {
            warnings.push(
                "stereo_correlation reports channel 0/1 pair only for multichannel material"
                    .to_string(),
            );
        }
        if sampled_pairs < total_pairs {
            warnings.push(format!(
                "inter-channel ITD/IACC metrics sampled {sampled_pairs}/{total_pairs} pairs"
            ));
        }

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
            inter_channel_correlation_matrix: corr_matrix,
            inter_channel_correlation_mean_abs: corr_mean_abs,
            inter_channel_correlation_min_abs: corr_min_abs,
            arrival_min_ms,
            arrival_max_ms,
            arrival_spread_ms,
            itd_01_ms,
            iacc_early_01,
            inter_channel_itd_mean_abs_ms,
            inter_channel_itd_max_abs_ms,
            inter_channel_iacc_early_mean,
            inter_channel_iacc_early_min,
            front_energy_ratio,
            rear_energy_ratio,
            height_energy_ratio,
            lfe_energy_ratio,
            warnings,
        }
    }
}

fn directional_energy_ratios(
    channels: &[Vec<f32>],
    map: &ChannelMap,
) -> anyhow::Result<(Option<f32>, Option<f32>, Option<f32>, Option<f32>)> {
    if channels.is_empty() || map.channels.is_empty() {
        return Ok((None, None, None, None));
    }

    let mut front = 0.0f32;
    let mut rear = 0.0f32;
    let mut height = 0.0f32;
    let mut lfe = 0.0f32;
    let mut total = 0.0f32;

    for entry in &map.channels {
        let ch = channels
            .get(entry.index)
            .ok_or_else(|| anyhow::anyhow!("channel map index out of bounds: {}", entry.index))?;
        let e = ch.iter().map(|x| x * x).sum::<f32>();
        total += e;

        if entry.is_lfe {
            lfe += e;
        } else if entry.elevation_deg.abs() >= 20.0 {
            height += e;
        } else if entry.azimuth_deg.abs() <= 45.0 {
            front += e;
        } else {
            rear += e;
        }
    }

    if total <= 1e-12 {
        return Ok((None, None, None, None));
    }

    Ok((
        Some(front / total),
        Some(rear / total),
        Some(height / total),
        Some(lfe / total),
    ))
}

fn inter_channel_correlation(
    channels: &[Vec<f32>],
) -> (Option<Vec<Vec<f32>>>, Option<f32>, Option<f32>) {
    let c = channels.len();
    if c < 2 {
        return (None, None, None);
    }

    let mut matrix = vec![vec![0.0f32; c]; c];
    let mut off_diag_abs = Vec::new();

    for i in 0..c {
        for j in i..c {
            let corr = if i == j {
                1.0
            } else {
                correlation(&channels[i], &channels[j])
            };
            matrix[i][j] = corr;
            matrix[j][i] = corr;
            if i != j {
                off_diag_abs.push(corr.abs());
            }
        }
    }

    if off_diag_abs.is_empty() {
        return (Some(matrix), None, None);
    }

    let mean = off_diag_abs.iter().sum::<f32>() / off_diag_abs.len() as f32;
    let min = off_diag_abs
        .iter()
        .fold(f32::INFINITY, |m, &v| if v < m { v } else { m });

    (Some(matrix), Some(mean), Some(min))
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
        num / (da.sqrt() * db.sqrt())
    }
}

fn first_arrival_sample(ir: &[f32]) -> Option<usize> {
    if ir.is_empty() {
        return None;
    }
    let peak = ir.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 1e-9 {
        return None;
    }
    let thresh = peak * 0.1;
    ir.iter().position(|x| x.abs() >= thresh)
}

fn arrival_spread_metrics(
    channels: &[Vec<f32>],
    sample_rate: u32,
) -> (Option<f32>, Option<f32>, Option<f32>) {
    let arrivals: Vec<f32> = channels
        .iter()
        .filter_map(|ch| first_arrival_sample(ch).map(|i| i as f32 * 1000.0 / sample_rate as f32))
        .collect();
    if arrivals.is_empty() {
        return (None, None, None);
    }
    let min_ms = arrivals
        .iter()
        .fold(f32::INFINITY, |m, &v| if v < m { v } else { m });
    let max_ms = arrivals
        .iter()
        .fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    let mean = arrivals.iter().sum::<f32>() / arrivals.len() as f32;
    let var = arrivals
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / arrivals.len() as f32;
    (Some(min_ms), Some(max_ms), Some(var.sqrt()))
}

fn estimate_itd_iacc(
    a: &[f32],
    b: &[f32],
    sample_rate: u32,
    early_window_s: f32,
    lag_window_s: f32,
) -> Option<(f32, f32)> {
    let n = a.len().min(b.len());
    if n < 32 {
        return None;
    }
    let window = ((early_window_s * sample_rate as f32).round() as usize).clamp(32, n);
    let max_lag = ((lag_window_s * sample_rate as f32).round() as isize).max(1);

    let mut best_lag = 0isize;
    let mut best_corr_abs = 0.0f32;

    for lag in -max_lag..=max_lag {
        let lag_abs = lag.unsigned_abs();
        if lag_abs >= window {
            continue;
        }
        let len = window - lag_abs;
        if len < 16 {
            continue;
        }
        let (sa, sb) = if lag >= 0 {
            (&a[lag as usize..lag as usize + len], &b[..len])
        } else {
            (&a[..len], &b[(-lag) as usize..(-lag) as usize + len])
        };
        let c = correlation(sa, sb).abs();
        if c > best_corr_abs {
            best_corr_abs = c;
            best_lag = lag;
        }
    }

    Some((best_lag as f32 * 1000.0 / sample_rate as f32, best_corr_abs))
}

#[allow(clippy::type_complexity)]
fn inter_channel_itd_iacc_metrics(
    channels: &[Vec<f32>],
    sample_rate: u32,
    early_window_s: f32,
    lag_window_s: f32,
    max_pairs: usize,
) -> (
    Option<f32>,
    Option<f32>,
    Option<f32>,
    Option<f32>,
    usize,
    usize,
) {
    let c = channels.len();
    if c < 2 {
        return (None, None, None, None, 0, 0);
    }
    let total_pairs = c * (c - 1) / 2;
    let mut sampled = 0usize;
    let mut itd_abs = Vec::new();
    let mut iacc = Vec::new();

    'pairs: for i in 0..c {
        for j in (i + 1)..c {
            // O(N^2) pair walks are correct and painful; cap early for large layouts.
            if sampled >= max_pairs {
                break 'pairs;
            }
            if let Some((lag_ms, iacc_v)) = estimate_itd_iacc(
                &channels[i],
                &channels[j],
                sample_rate,
                early_window_s,
                lag_window_s,
            ) {
                itd_abs.push(lag_ms.abs());
                iacc.push(iacc_v);
                sampled += 1;
            }
        }
    }

    if sampled == 0 {
        return (None, None, None, None, 0, total_pairs);
    }

    let itd_mean = itd_abs.iter().sum::<f32>() / itd_abs.len() as f32;
    let itd_max = itd_abs
        .iter()
        .fold(0.0f32, |m, &v| if v > m { v } else { m });
    let iacc_mean = iacc.iter().sum::<f32>() / iacc.len() as f32;
    let iacc_min = iacc
        .iter()
        .fold(f32::INFINITY, |m, &v| if v < m { v } else { m });

    (
        Some(itd_mean),
        Some(itd_max),
        Some(iacc_mean),
        Some(iacc_min),
        sampled,
        total_pairs,
    )
}
