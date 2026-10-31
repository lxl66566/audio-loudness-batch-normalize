//! Per-file normalization: gain decision, decode, gain application, and save.

use std::{
    fs,
    path::{Path, PathBuf},
};

use log::{debug, info, warn};
use pathdiff::diff_paths;

use crate::{
    decode,
    error::{Error, MeasurementError, ProcessingError},
    format::AudioFormat,
    measurement::{AudioStats, LoudnessMeasurements, measure_single_file_stats},
    save::{EncodeConfig, save_ogg, save_wav},
};

/// Outcome of processing a single file, used to build the run report.
#[derive(Debug, Clone)]
pub struct ProcessedFile {
    /// Input file path.
    pub path: PathBuf,
    /// Path of the file written by this run; `None` when the file was
    /// skipped (silent or unmeasurable input).
    pub output_path: Option<PathBuf>,
    /// Measured statistics of the input; `None` when skipped.
    pub stats: Option<AudioStats>,
    /// Final applied gain in dB (0.0 when skipped).
    pub gain_db: f64,
    /// Whether true-peak limiting reduced the gain below the loudness gain.
    pub peak_limited: bool,
}

/// Combined gain decision for a single file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GainDecision {
    /// Combined linear gain.
    pub(crate) linear: f64,
    /// Combined gain in dB.
    pub(crate) db: f64,
    /// Whether true-peak limiting reduced the gain below the loudness gain.
    pub(crate) peak_limited: bool,
}

/// Computes the combined gain that moves the loudness of `stats` to
/// `target_lufs`, additionally limited so the true peak after the gain does
/// not exceed `target_peak_db`.
///
/// The measurement is linear, so a constant gain shifts the true peak by the
/// same dB amount.
pub(crate) fn compute_gain(
    stats: AudioStats,
    target_lufs: f64,
    target_peak_db: f64,
) -> GainDecision {
    let loudness_gain_db = target_lufs - stats.loudness_lufs;
    let peak_after_gain_db = stats.true_peak_db + loudness_gain_db;
    let peak_limiting_gain_db = (target_peak_db - peak_after_gain_db).min(0.0);
    let db = loudness_gain_db + peak_limiting_gain_db;
    GainDecision {
        linear: 10.0_f64.powf(db / 20.0),
        db,
        peak_limited: peak_limiting_gain_db < 0.0,
    }
}

/// Main orchestration function for processing a single file using the two-pass
/// method: measurement (or reuse of the cached sampling measurement) followed
/// by the gain pass.
///
/// Returns the outcome of the file for the run report.
// The cache is built internally with the default hasher; generalizing it over
// `BuildHasher` would leak a type parameter into every call site.
#[allow(clippy::implicit_hasher)]
pub fn process_single_file(
    input_path: impl AsRef<Path>,
    target_lufs: f64,
    target_peak_db: f64,
    input_base_dir: impl AsRef<Path>,
    output_base_dir: &Option<impl AsRef<Path>>,
    cache: &Option<LoudnessMeasurements>,
    encode: &EncodeConfig,
) -> Result<ProcessedFile, Error> {
    let input_path = input_path.as_ref();
    let file_format = AudioFormat::from_path(input_path).ok_or_else(|| Error::Processing {
        path: input_path.to_path_buf(),
        source: ProcessingError::UnsupportedFormat,
    })?;
    let file_name_str = input_path.file_name().unwrap_or_default().to_string_lossy();
    debug!("Processing: {file_name_str}");

    // --- MEASUREMENT ---

    // 1. Measure loudness and true peak in a single decode pass (or reuse the
    // cached measurement from the sampling phase)
    let stats = match cache.as_ref().and_then(|x| x.get(input_path)) {
        Some(Ok(stats)) => *stats,
        Some(Err(_)) => {
            return Err(Error::Measurement {
                path: input_path.to_path_buf(),
                source: MeasurementError::SamplingFailed,
            });
        },
        None => measure_single_file_stats(input_path).map_err(|e| Error::Measurement {
            path: input_path.to_path_buf(),
            source: e,
        })?,
    };

    let current_lufs = stats.loudness_lufs;

    // Handle silence or measurement errors
    if !current_lufs.is_finite() {
        let reason = if current_lufs.is_infinite() && current_lufs.is_sign_negative() {
            "silent file"
        } else {
            "non-finite loudness"
        };
        info!("Skipping processing for {reason}: {file_name_str}");
        return Ok(ProcessedFile {
            path: input_path.to_path_buf(),
            output_path: None,
            stats: None,
            gain_db: 0.0,
            peak_limited: false,
        });
    }

    // 2. Calculate the final, combined linear gain
    let gain = compute_gain(stats, target_lufs, target_peak_db);

    debug!(
        "  -> File: {file_name_str}, Current LUFS: {current_lufs:.2}, Target LUFS: \
         {target_lufs:.2}, Final Combined Linear Gain: {:.3}x",
        gain.linear
    );

    // --- PROCESSING & SAVING ---

    // Determine output path
    let mut output_path = match output_base_dir.as_ref() {
        Some(obd) => {
            let relative_path =
                diff_paths(input_path, input_base_dir).ok_or_else(|| Error::Io {
                    path: input_path.to_path_buf(),
                    source: std::io::Error::other("Failed to calculate relative path"),
                })?;
            obd.as_ref().join(relative_path)
        },
        None => input_path.to_path_buf(),
    };

    // For non-WAV formats that will be converted, change the extension.
    if !matches!(file_format, AudioFormat::Wav | AudioFormat::Ogg) {
        output_path.set_extension("wav");
        if output_base_dir.is_none() {
            warn!(
                "In-place mode: {file_name_str} is converted to WAV, but the original file is \
                 kept unnormalized and will be processed again on the next run."
            );
        }
    }

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    // Decode, apply final gain, and save in a streaming manner
    decode_apply_gain_and_save(input_path, &output_path, &file_format, gain.linear, encode)?;

    debug!(
        "Successfully wrote normalized file to {}",
        output_path.display()
    );
    Ok(ProcessedFile {
        path: input_path.to_path_buf(),
        output_path: Some(output_path),
        stats: Some(stats),
        gain_db: gain.db,
        peak_limited: gain.peak_limited,
    })
}

/// Decodes the audio file again, applies the final calculated gain, and saves
/// the result. The output is written to a temporary file first and atomically
/// renamed into place, so even an in-place overwrite never truncates the input
/// before the new content is complete.
fn decode_apply_gain_and_save(
    input_path: &Path,
    output_path: &Path,
    file_format: &AudioFormat,
    final_linear_gain: f64,
    encode: &EncodeConfig,
) -> Result<(), Error> {
    let audio = decode::open(input_path).map_err(|e| Error::Processing {
        path: input_path.to_path_buf(),
        source: e.into(),
    })?;

    match file_format {
        AudioFormat::Ogg => save_ogg(
            audio,
            input_path,
            output_path,
            final_linear_gain,
            encode.ogg_quality,
        ),
        // All other formats are saved as WAV.
        _ => save_wav(
            audio,
            input_path,
            output_path,
            final_linear_gain,
            encode.wav_bit_depth,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::compute_gain;
    use crate::measurement::AudioStats;

    #[test]
    fn gain_without_peak_limiting() {
        let stats = AudioStats {
            loudness_lufs: -20.0,
            true_peak_db: -12.0,
        };
        // +6 dB loudness gain, peak after gain -6.0 dBTP < -1.5 dBTP: no
        // limiting.
        let gain = compute_gain(stats, -14.0, -1.5);
        assert!((gain.linear - 10.0_f64.powf(6.0 / 20.0)).abs() < 1e-12);
        assert!((gain.db - 6.0).abs() < 1e-9);
        assert!(!gain.peak_limited);
    }

    #[test]
    fn gain_limited_by_true_peak() {
        let stats = AudioStats {
            loudness_lufs: -20.0,
            true_peak_db: -5.0,
        };
        // +6 dB loudness gain would push the peak to +1.0 dBTP; limited to
        // -1.5 dBTP, so the combined gain is 6.0 - 2.5 = 3.5 dB.
        let gain = compute_gain(stats, -14.0, -1.5);
        assert!((gain.linear - 10.0_f64.powf(3.5 / 20.0)).abs() < 1e-12);
        assert!((gain.db - 3.5).abs() < 1e-9);
        assert!(gain.peak_limited);
    }
}
