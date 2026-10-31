//! Loudness and true-peak measurement of audio files.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use ebur128::{EbuR128, Mode};
use log::{debug, warn};

use crate::{
    decode,
    error::{MeasurementError, ProcessingError},
};

/// Loudness statistics of a single audio file, measured in one decode pass
#[derive(Debug, Clone, Copy)]
pub struct AudioStats {
    /// Integrated loudness in LUFS
    pub loudness_lufs: f64,
    /// True peak in dBTP
    pub true_peak_db: f64,
}

/// Measurement results keyed by file path; failures are kept so later stages
/// can report them.
pub type LoudnessMeasurements = HashMap<PathBuf, Result<AudioStats, MeasurementError>>;

/// Measures the loudness and true peak of a single audio file in one decode
/// pass, using EBU R128 with an RMS fallback for the loudness
pub fn measure_single_file_stats(path: impl AsRef<Path>) -> Result<AudioStats, MeasurementError> {
    let path = path.as_ref();
    let mut audio = decode::open(path)?;
    let mut ebu_state = EbuR128::new(
        u32::from(audio.channels),
        audio.sample_rate,
        Mode::I | Mode::TRUE_PEAK,
    )?;

    let mut rms_sum_of_squares = 0.0;
    let mut rms_total_samples = 0u64;

    audio.for_each_packet(|planar| {
        let plane_slices: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
        if let Err(e) = ebu_state.add_frames_planar_f32(&plane_slices) {
            warn!("EBU R128 add_frames failed: {e}. Skipping chunk.");
        }
        update_rms_accumulators(planar, &mut rms_sum_of_squares, &mut rms_total_samples);
        Ok::<(), MeasurementError>(())
    })?;

    let loudness_lufs = match ebu_state.loudness_global() {
        Ok(lufs) if lufs.is_finite() => lufs,
        _ => {
            debug!("EBU R128 failed or gave non-finite value. Falling back to RMS.");
            calculate_rms_dbfs(rms_sum_of_squares, rms_total_samples)
        },
    };

    // Maximum true peak across all channels. `EbuR128::true_peak` returns a
    // linear amplitude; convert it to dBTP. If the EBU R128 state is broken,
    // fall back to negative infinity (no peak limiting will be applied).
    let true_peak_db = (0..u32::from(audio.channels))
        .map(|i| ebu_state.true_peak(i))
        .collect::<Result<Vec<_>, _>>()
        .map_or(f64::NEG_INFINITY, |peaks| {
            let linear = peaks.into_iter().fold(0.0_f64, f64::max);
            if linear > 0.0 { 20.0 * linear.log10() } else { f64::NEG_INFINITY }
        });

    Ok(AudioStats {
        loudness_lufs,
        true_peak_db,
    })
}

/// Measures the loudness of a single audio file using EBU R128 or RMS fallback
pub fn measure_single_file_loudness(path: impl AsRef<Path>) -> Result<f64, MeasurementError> {
    Ok(measure_single_file_stats(path)?.loudness_lufs)
}

fn update_rms_accumulators(
    planar_f32: &[Vec<f32>],
    sum_of_squares: &mut f64,
    total_samples: &mut u64,
) {
    for channel_buffer in planar_f32 {
        for sample in channel_buffer {
            *sum_of_squares += f64::from(*sample) * f64::from(*sample);
        }
        *total_samples += channel_buffer.len() as u64;
    }
}

fn calculate_rms_dbfs(sum_of_squares: f64, total_samples: u64) -> f64 {
    if total_samples == 0 || sum_of_squares <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mean_square = sum_of_squares / total_samples as f64;
    20.0 * mean_square.sqrt().log10()
}

/// Computes the target loudness as the mean of the valid measurements, with
/// `trim_percentage` of them removed from each end.
pub(crate) fn calculate_target_loudness(
    measurements: &LoudnessMeasurements,
    trim_percentage: f64,
) -> Result<f64, ProcessingError> {
    let mut valid_loudnesses: Vec<f64> = measurements
        .values()
        .filter_map(|r| {
            r.as_ref()
                .ok()
                .map(|stats| stats.loudness_lufs)
                .filter(|l| l.is_finite())
        })
        .collect();

    if valid_loudnesses.is_empty() {
        return Err(ProcessingError::TargetLoudnessCalculationFailed(
            "No valid finite loudness measurements available.".to_string(),
        ));
    }

    valid_loudnesses.sort_by(f64::total_cmp);
    let count = valid_loudnesses.len();
    let trim_count = (count as f64 * trim_percentage).floor() as usize;

    let trimmed_slice = if count > trim_count * 2 {
        &valid_loudnesses[trim_count..count - trim_count]
    } else {
        warn!("Not enough samples to perform trimming. Using mean of all valid samples.");
        &valid_loudnesses[..]
    };

    if trimmed_slice.is_empty() {
        return Err(ProcessingError::TargetLoudnessCalculationFailed(
            "Trimmed slice is empty.".to_string(),
        ));
    }

    let trimmed_slice_len = trimmed_slice.len();
    let mean: f64 = trimmed_slice.iter().sum::<f64>() / trimmed_slice_len as f64;
    if !mean.is_finite() {
        return Err(ProcessingError::TargetLoudnessCalculationFailed(format!(
            "Calculated target loudness is not a finite number ({mean:.2})."
        )));
    }
    Ok(mean)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use super::{AudioStats, LoudnessMeasurements, calculate_rms_dbfs, calculate_target_loudness};

    fn measurement_map(loudnesses: &[f64]) -> LoudnessMeasurements {
        loudnesses
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                (
                    PathBuf::from(format!("{i}")),
                    Ok(AudioStats {
                        loudness_lufs: l,
                        true_peak_db: 0.0,
                    }),
                )
            })
            .collect()
    }

    #[test]
    fn target_loudness_trims_both_ends() {
        let measurements = measurement_map(&[-30.0, -20.0, -10.0, -10.0, 0.0]);
        // Trim 1 value from each end: mean of [-20, -10, -10].
        assert!(
            (calculate_target_loudness(&measurements, 0.2).unwrap() - (-40.0 / 3.0)).abs() < 1e-9
        );
    }

    #[test]
    fn target_loudness_falls_back_when_trim_would_empty_the_slice() {
        // count == 2 * trim_count: not enough samples to trim, mean of all.
        let measurements = measurement_map(&[-40.0, -30.0, -20.0, -10.0]);
        assert!((calculate_target_loudness(&measurements, 0.5).unwrap() - (-25.0)).abs() < 1e-9);
    }

    #[test]
    fn target_loudness_ignores_non_finite_measurements() {
        let measurements = measurement_map(&[f64::NAN, -20.0, f64::INFINITY, -10.0]);
        assert!((calculate_target_loudness(&measurements, 0.0).unwrap() - (-15.0)).abs() < 1e-9);
    }

    #[test]
    fn target_loudness_fails_without_valid_measurements() {
        let measurements = measurement_map(&[f64::NAN, f64::INFINITY]);
        assert!(calculate_target_loudness(&measurements, 0.3).is_err());
        assert!(calculate_target_loudness(&HashMap::new(), 0.3).is_err());
    }

    #[test]
    fn rms_dbfs_of_empty_or_silent_signal() {
        assert_eq!(calculate_rms_dbfs(0.0, 0), f64::NEG_INFINITY);
        assert_eq!(calculate_rms_dbfs(0.0, 100), f64::NEG_INFINITY);
    }

    #[test]
    fn rms_dbfs_of_known_amplitude() {
        // 4 samples at amplitude 0.5 -> RMS 0.5 -> -6.0206 dBFS.
        assert!((calculate_rms_dbfs(4.0 * 0.25, 4) - 20.0 * 0.5_f64.log10()).abs() < 1e-9);
    }
}
