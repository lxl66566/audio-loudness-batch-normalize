#![allow(clippy::needless_range_loop)]

/// Module for opening and decoding audio files
pub mod decode;
/// Module for error handling
pub mod error;
/// Module for saving audio files
pub mod save;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use ebur128::{EbuR128, Mode};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use rand::seq::IndexedRandom as _;
use rayon::prelude::*;
use strum::IntoEnumIterator as _;
use strum_macros::{Display, EnumIter};
use walkdir::WalkDir;

use crate::{
    error::{Error, MeasurementError, ProcessingError},
    save::{save_ogg, save_wav},
};

/// Represents a supported audio file format. The strum serialization of each
/// variant is the corresponding file extension.
#[derive(Debug, PartialEq, Display, Clone, EnumIter)]
#[strum(serialize_all = "camelCase")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    M4a,
    Aac,
    Opus,
}

impl AudioFormat {
    /// Creates an AudioFormat from a file path based on its extension
    /// (case-insensitive).
    #[inline]
    pub fn from_path(value: impl AsRef<Path>) -> Option<Self> {
        let ext = value.as_ref().extension()?.to_string_lossy().to_lowercase();
        Self::iter().find(|format| format.to_string() == ext)
    }
}

/// Configuration options for audio normalization process
///
/// With the `bin` feature the struct doubles as the clap argument definition
/// of the command-line interface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bin", derive(clap::Args))]
pub struct NormalizationOptions {
    /// input directory
    #[cfg_attr(feature = "bin", arg(index = 1))]
    pub input_dir: PathBuf,
    /// output directory, default to override input audios
    #[cfg_attr(feature = "bin", arg(short = 'o', long = "output"))]
    pub output_dir: Option<PathBuf>,
    /// percentage of files to sample for calculating target loudness (0.0 to
    /// 1.0), default to sample all files
    #[cfg_attr(
        feature = "bin",
        arg(short, long, default_value_t = Self::DEFAULT_SAMPLE_PERCENTAGE)
    )]
    pub sample_percentage: f64,
    /// trim percentage for calculating target loudness (0.0 to 0.5)
    #[cfg_attr(
        feature = "bin",
        arg(long, default_value_t = Self::DEFAULT_TRIM_PERCENTAGE)
    )]
    pub trim_percentage: f64,
    /// target loudness in LUFS, default to automatic calculation
    #[cfg_attr(feature = "bin", arg(long))]
    pub target_lufs: Option<f64>,
    /// target true peak in dBTP
    #[cfg_attr(
        feature = "bin",
        arg(long, default_value_t = Self::DEFAULT_TRUE_PEAK_DB)
    )]
    pub true_peak_db: f64,
    /// number of threads to use, default to CPU core count
    #[cfg_attr(feature = "bin", arg(short = 't', long = "threads"))]
    pub num_threads: Option<usize>,
    /// only measure and report the target loudness (LUFS) without writing
    /// normalized files
    #[cfg_attr(feature = "bin", arg(short = 'n', long))]
    pub measure_only: bool,
}

impl NormalizationOptions {
    /// Default percentage of files to sample for target loudness calculation.
    pub const DEFAULT_SAMPLE_PERCENTAGE: f64 = 1.0;
    /// Default percentage of measurements to trim from each end.
    pub const DEFAULT_TRIM_PERCENTAGE: f64 = 0.3;
    /// Default target true peak in dBTP.
    pub const DEFAULT_TRUE_PEAK_DB: f64 = -1.5;
}

impl Default for NormalizationOptions {
    fn default() -> Self {
        Self {
            input_dir: PathBuf::from("."),
            output_dir: None,
            sample_percentage: Self::DEFAULT_SAMPLE_PERCENTAGE,
            trim_percentage: Self::DEFAULT_TRIM_PERCENTAGE,
            target_lufs: None,
            true_peak_db: Self::DEFAULT_TRUE_PEAK_DB,
            num_threads: None,
            measure_only: false,
        }
    }
}

/// Loudness statistics of a single audio file, measured in one decode pass
#[derive(Debug, Clone, Copy)]
pub struct AudioStats {
    /// Integrated loudness in LUFS
    pub loudness_lufs: f64,
    /// True peak in dBTP
    pub true_peak_db: f64,
}

/// Shared progress bar style for the parallel stages.
const PROGRESS_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}";

fn progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(PROGRESS_TEMPLATE)
        .expect("internal error: invalid progress bar template")
        .progress_chars("#>-")
}

/// Normalize the loudness of all audio files in a folder
pub fn normalize_folder_loudness(options: &NormalizationOptions) -> Result<(), Error> {
    // 1. Validate options
    validate_options(options)?;

    // 2. Discover audio files
    info!("Discovering audio files in {:?}...", options.input_dir);
    let all_audio_files = find_audio_files(&options.input_dir)?;
    if all_audio_files.is_empty() {
        info!("No audio files found.");
        return Ok(());
    }
    info!("Found {} audio files.", all_audio_files.len());

    // A local pool avoids mutating the global rayon pool; num_threads == 0
    // falls back to rayon's default thread count.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.num_threads.unwrap_or(0))
        .build()
        .map_err(|e| Error::InvalidOptions(format!("Failed to build thread pool: {e}")))?;
    if let Some(n) = options.num_threads.filter(|&n| n > 0) {
        info!("Using {n} threads for processing.");
    } else {
        info!("Using default number of threads.");
    }

    // --- Target Loudness Calculation ---
    let (target_lufs, loudness_measurements_cache) = match options.target_lufs {
        Some(t) => {
            info!("Using user-provided Target Loudness: {t:.2} LUFS");
            (t, None)
        }
        None => {
            // 3. Select sample files
            let sample_size =
                (all_audio_files.len() as f64 * options.sample_percentage).ceil() as usize;
            if sample_size == 0 {
                return Err(Error::InvalidOptions(
                    "Sample size is 0, please check your sample percentage.".to_string(),
                ));
            }
            let sampled_files: Vec<&Path> = if sample_size >= all_audio_files.len() {
                all_audio_files.iter().map(PathBuf::as_path).collect()
            } else {
                let mut rng = rand::rng();
                all_audio_files
                    .sample(&mut rng, sample_size)
                    .map(PathBuf::as_path)
                    .collect()
            };
            info!(
                "Selected {} files for target loudness calculation.",
                sampled_files.len()
            );

            // 4. Measure loudness of sampled files (in parallel)
            info!("Measuring loudness of sampled files...");
            let sample_pb = ProgressBar::new(sampled_files.len() as u64);
            sample_pb.set_style(progress_style());
            sample_pb.set_message("Measuring samples");

            let loudness_measurements: HashMap<PathBuf, Result<AudioStats, MeasurementError>> =
                pool.install(|| {
                    sampled_files
                        .par_iter()
                        .progress_with(sample_pb.clone())
                        .map(|path| {
                            let loudness_result = measure_single_file_stats(path);
                            if let Err(e) = &loudness_result {
                                warn!(
                                    "Failed to measure loudness for sample {:?}: {}",
                                    path.file_name().unwrap_or_default(),
                                    e
                                );
                            }
                            (path.to_path_buf(), loudness_result)
                        })
                        .collect()
                });
            sample_pb.finish_with_message("Sample measurement done");

            // 5. Calculate target loudness (trimmed mean)
            let calculated_target =
                calculate_target_loudness(&loudness_measurements, options.trim_percentage)
                    .map_err(|e| Error::Processing {
                        path: options.input_dir.to_path_buf(),
                        source: e,
                    })?;
            info!(
                "Calculated Target Loudness ({}% trimmed mean): {:.2} LUFS",
                options.trim_percentage * 100.0,
                calculated_target
            );

            (calculated_target, Some(loudness_measurements))
        }
    };

    if options.measure_only {
        info!(
            "Measure-only mode: target loudness is {target_lufs:.2} LUFS. Skipping normalization."
        );
        return Ok(());
    }

    // --- Processing All Files ---
    info!(
        "Processing all {} files to target {:.2} LUFS / {:.1} dBTP...",
        all_audio_files.len(),
        target_lufs,
        options.true_peak_db
    );
    let process_pb = ProgressBar::new(all_audio_files.len() as u64);
    process_pb.set_style(progress_style());
    process_pb.set_message("Processing files");

    let results: Vec<Result<(), Error>> = pool.install(|| {
        all_audio_files
            .par_iter()
            .progress_with(process_pb.clone())
            .map(|path| {
                process_single_file(
                    path,
                    target_lufs,
                    options.true_peak_db,
                    &options.input_dir,
                    &options.output_dir,
                    &loudness_measurements_cache,
                )
            })
            .collect()
    });
    process_pb.finish_with_message("Processing done");

    // 7. Report final status and errors
    let mut success_count = 0;
    let mut error_count = 0;
    for result in results {
        match result {
            Ok(_) => success_count += 1,
            Err(e) => {
                error!("Error: {e}"); // Log the detailed error
                error_count += 1;
            }
        }
    }

    info!("Processing complete. {success_count} files succeeded, {error_count} files failed.");

    if error_count > 0 {
        Err(Error::Processing {
            path: options.input_dir.to_path_buf(),
            source: ProcessingError::FilesFailed(error_count),
        })
    } else {
        Ok(())
    }
}

/// Main orchestration function for processing a single file using the two-pass
/// method.
pub fn process_single_file(
    input_path: impl AsRef<Path>,
    target_lufs: f64,
    target_peak_db: f64,
    input_base_dir: impl AsRef<Path>,
    output_base_dir: &Option<impl AsRef<Path>>,
    cache: &Option<HashMap<PathBuf, Result<AudioStats, MeasurementError>>>,
) -> Result<(), Error> {
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
        }
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
        return Ok(());
    }

    // 2. Calculate the final, combined linear gain
    let final_linear_gain = compute_gain(stats, target_lufs, target_peak_db);

    debug!(
        "  -> File: {file_name_str}, Current LUFS: {current_lufs:.2}, Target LUFS: {target_lufs:.2}, Final Combined Linear Gain: {final_linear_gain:.3}x"
    );

    // --- PROCESSING & SAVING ---

    // Determine output path
    let mut output_path = match output_base_dir.as_ref() {
        Some(obd) => {
            let relative_path =
                pathdiff::diff_paths(input_path, input_base_dir).ok_or_else(|| Error::Io {
                    path: input_path.to_path_buf(),
                    source: std::io::Error::other("Failed to calculate relative path"),
                })?;
            obd.as_ref().join(relative_path)
        }
        None => input_path.to_path_buf(),
    };

    // For non-WAV formats that will be converted, change the extension.
    if !matches!(file_format, AudioFormat::Wav | AudioFormat::Ogg) {
        output_path.set_extension("wav");
        if output_base_dir.is_none() {
            warn!(
                "In-place mode: {file_name_str} is converted to WAV, but the original file is kept unnormalized and will be processed again on the next run."
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
    decode_apply_gain_and_save(input_path, &output_path, &file_format, final_linear_gain)?;

    debug!("Successfully wrote normalized file to {output_path:?}");
    Ok(())
}

/// Computes the combined linear gain that moves the loudness of `stats` to
/// `target_lufs`, additionally limited so the true peak after the gain does
/// not exceed `target_peak_db`.
///
/// The measurement is linear, so a constant gain shifts the true peak by the
/// same dB amount.
fn compute_gain(stats: AudioStats, target_lufs: f64, target_peak_db: f64) -> f64 {
    let loudness_gain_db = target_lufs - stats.loudness_lufs;
    let peak_after_gain_db = stats.true_peak_db + loudness_gain_db;
    let peak_limiting_gain_db = (target_peak_db - peak_after_gain_db).min(0.0);
    10.0_f64.powf((loudness_gain_db + peak_limiting_gain_db) / 20.0)
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
) -> Result<(), Error> {
    let audio = decode::open(input_path).map_err(|e| Error::Processing {
        path: input_path.to_path_buf(),
        source: e.into(),
    })?;

    match file_format {
        AudioFormat::Ogg => save_ogg(audio, input_path, output_path, final_linear_gain),
        // All other formats are saved as WAV.
        _ => save_wav(audio, input_path, output_path, final_linear_gain),
    }
}

/// Measures the loudness and true peak of a single audio file in one decode
/// pass, using EBU R128 with an RMS fallback for the loudness
pub fn measure_single_file_stats(path: impl AsRef<Path>) -> Result<AudioStats, MeasurementError> {
    let path = path.as_ref();
    let mut audio = decode::open(path)?;
    let mut ebu_state = EbuR128::new(
        audio.channels as u32,
        audio.sample_rate,
        Mode::I | Mode::TRUE_PEAK,
    )?;

    let mut rms_sum_of_squares = 0.0;
    let mut rms_total_samples = 0u64;

    audio.for_each_packet(|planar| {
        let plane_slices: Vec<&[f32]> = planar.iter().map(|v| v.as_slice()).collect();
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
        }
    };

    // Maximum true peak across all channels. `EbuR128::true_peak` returns a
    // linear amplitude; convert it to dBTP. If the EBU R128 state is broken,
    // fall back to negative infinity (no peak limiting will be applied).
    let channel_count = audio.channels as usize;
    let true_peak_db = (0..channel_count)
        .map(|i| ebu_state.true_peak(i as u32))
        .collect::<Result<Vec<_>, _>>()
        .map(|peaks| {
            let linear = peaks.into_iter().fold(0.0_f64, f64::max);
            if linear > 0.0 {
                20.0 * linear.log10()
            } else {
                f64::NEG_INFINITY
            }
        })
        .unwrap_or(f64::NEG_INFINITY);

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
            *sum_of_squares += (*sample as f64) * (*sample as f64);
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

fn calculate_target_loudness(
    measurements: &HashMap<PathBuf, Result<AudioStats, MeasurementError>>,
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

/// Validates normalization options for correctness
///
/// # Arguments
/// * `options` - Reference to NormalizationOptions struct
fn validate_options(options: &NormalizationOptions) -> Result<(), Error> {
    if !options.input_dir.is_dir() {
        return Err(Error::InvalidOptions(format!(
            "Input path is not a valid directory: {:?}",
            options.input_dir
        )));
    }
    if let Some(output_dir) = &options.output_dir {
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).map_err(|e| Error::Io {
                path: output_dir.to_path_buf(),
                source: e,
            })?;
            info!("Created output directory: {output_dir:?}");
        } else if !output_dir.is_dir() {
            return Err(Error::InvalidOptions(format!(
                "Output path exists but is not a directory: {output_dir:?}"
            )));
        }
    }

    if !(0.0..=1.0).contains(&options.sample_percentage) || options.sample_percentage == 0.0 {
        return Err(Error::InvalidOptions(format!(
            "Sample percentage must be in the range (0.0, 1.0]: {}",
            options.sample_percentage
        )));
    }
    if !(0.0..0.5).contains(&options.trim_percentage) {
        return Err(Error::InvalidOptions(format!(
            "Trim percentage must be between 0.0 and 0.5 (exclusive of 0.5): {}",
            options.trim_percentage
        )));
    }
    if options.true_peak_db > 0.0 {
        warn!(
            "Target true peak {:.1} dBTP is above 0 dBFS. This will likely cause clipping in standard formats.",
            options.true_peak_db
        );
    }
    Ok(())
}

/// Finds all supported audio files in the specified directory
///
/// # Arguments
/// * `input_dir` - Directory to search for audio files
///
/// # Returns
/// Vector of paths to the found audio files
fn find_audio_files(input_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, Error> {
    let mut audio_files = Vec::new();

    for entry in WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok()) // Filter out directory reading errors
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if AudioFormat::from_path(path).is_some() {
            audio_files.push(path.to_path_buf());
        }
    }
    Ok(audio_files)
}

#[cfg(test)]
mod tests {
    use symphonia::core::audio::{
        AudioBuffer, AudioMut, AudioSpec, Channels, GenericAudioBufferRef, sample::i24,
    };

    use super::*;

    fn measurement_map(
        loudnesses: &[f64],
    ) -> HashMap<PathBuf, Result<AudioStats, MeasurementError>> {
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
    fn gain_without_peak_limiting() {
        let stats = AudioStats {
            loudness_lufs: -20.0,
            true_peak_db: -12.0,
        };
        // +6 dB loudness gain, peak after gain -6.0 dBTP < -1.5 dBTP: no
        // limiting.
        assert!((compute_gain(stats, -14.0, -1.5) - 10.0_f64.powf(6.0 / 20.0)).abs() < 1e-12);
    }

    #[test]
    fn gain_limited_by_true_peak() {
        let stats = AudioStats {
            loudness_lufs: -20.0,
            true_peak_db: -5.0,
        };
        // +6 dB loudness gain would push the peak to +1.0 dBTP; limited to
        // -1.5 dBTP, so the combined gain is 6.0 - 2.5 = 3.5 dB.
        assert!((compute_gain(stats, -14.0, -1.5) - 10.0_f64.powf(3.5 / 20.0)).abs() < 1e-12);
    }

    #[test]
    fn converts_u8_to_planar_f32() {
        let mut buf = AudioBuffer::<u8>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 3);
        buf.resize_uninit(3);
        buf.plane_mut(0).unwrap().copy_from_slice(&[128, 255, 0]);
        let planar =
            decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::U8(&buf)).unwrap();
        assert_eq!(planar.len(), 1);
        assert!((planar[0][0] - 0.0).abs() < 1e-6);
        assert!((planar[0][1] - 127.0 / 128.0).abs() < 1e-6);
        assert!((planar[0][2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn converts_s16_to_planar_f32() {
        let mut buf = AudioBuffer::<i16>::new(AudioSpec::new(44_100, Channels::Discrete(2)), 2);
        buf.resize_uninit(2);
        buf.plane_mut(0).unwrap().copy_from_slice(&[0, i16::MAX]);
        buf.plane_mut(1).unwrap().copy_from_slice(&[i16::MIN, -1]);
        let planar =
            decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::S16(&buf)).unwrap();
        assert_eq!(planar.len(), 2);
        assert_eq!(planar[0][0], 0.0);
        assert_eq!(planar[0][1], 1.0);
        assert_eq!(planar[1][0], i16::MIN as f32 / i16::MAX as f32);
        assert_eq!(planar[1][1], -1.0 / i16::MAX as f32);
    }

    #[test]
    fn converts_s24_to_planar_f32() {
        let mut buf = AudioBuffer::<i24>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        buf.resize_uninit(2);
        buf.plane_mut(0)
            .unwrap()
            .copy_from_slice(&[i24::MAX, i24::MIN]);
        let planar =
            decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::S24(&buf)).unwrap();
        assert_eq!(planar[0][0], 1.0);
        assert_eq!(planar[0][1], -8_388_608.0 / 8_388_607.0);
    }

    #[test]
    fn converts_s32_to_planar_f32() {
        let mut buf = AudioBuffer::<i32>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        buf.resize_uninit(2);
        buf.plane_mut(0).unwrap().copy_from_slice(&[0, i32::MAX]);
        let planar =
            decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::S32(&buf)).unwrap();
        assert_eq!(planar[0][0], 0.0);
        assert_eq!(planar[0][1], 1.0);
    }

    #[test]
    fn converts_floats_to_planar_f32() {
        let mut f32_buf = AudioBuffer::<f32>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        f32_buf.resize_uninit(2);
        f32_buf.plane_mut(0).unwrap().copy_from_slice(&[0.25, -0.5]);
        let planar =
            decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::F32(&f32_buf)).unwrap();
        assert_eq!(planar[0], vec![0.25, -0.5]);

        let mut f64_buf = AudioBuffer::<f64>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        f64_buf.resize_uninit(2);
        f64_buf.plane_mut(0).unwrap().copy_from_slice(&[0.25, -0.5]);
        let planar =
            decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::F64(&f64_buf)).unwrap();
        assert_eq!(planar[0], vec![0.25, -0.5]);
    }

    #[test]
    fn rejects_unsupported_buffer_format() {
        let mut buf = AudioBuffer::<u16>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 1);
        buf.resize_uninit(1);
        buf.plane_mut(0).unwrap().copy_from_slice(&[0u16]);
        assert!(decode::convert_buffer_to_planar_f32(&GenericAudioBufferRef::U16(&buf)).is_err());
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

    #[test]
    fn format_from_path_is_case_insensitive() {
        assert_eq!(AudioFormat::from_path("a/b/c.WAV"), Some(AudioFormat::Wav));
        assert_eq!(AudioFormat::from_path("x.Mp3"), Some(AudioFormat::Mp3));
        assert_eq!(AudioFormat::from_path("y.ogg"), Some(AudioFormat::Ogg));
    }

    #[test]
    fn format_from_path_rejects_unknown_extensions() {
        assert_eq!(AudioFormat::from_path("x.txt"), None);
        assert_eq!(AudioFormat::from_path("noext"), None);
        assert_eq!(AudioFormat::from_path(".hidden"), None);
    }
}
