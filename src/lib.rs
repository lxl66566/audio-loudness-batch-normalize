/// Module for error handling
pub mod error;
/// Module for saving audio files
pub mod save;

use crate::error::{Error, MeasurementError};
use crate::save::save_as_ogg;
use ebur128::{EbuR128, Mode};
use error::ProcessingError;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use rand::seq::IndexedRandom as _;
use rayon::prelude::*;
use save::save_as_wav;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use strum_macros::Display;
use symphonia::core::audio::{AudioBufferRef, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use walkdir::WalkDir;

/// Represents supported audio file formats
#[derive(Debug, PartialEq, Display)]
#[strum(serialize_all = "camelCase")]
pub enum AudioFormats {
    Wav,
    Mp3,
    Flac,
    Ogg,
    M4a,
    Aac,
    Opus,
}

impl AudioFormats {
    /// Returns a list of supported file extensions
    #[inline]
    pub fn supported_extensions() -> &'static [&'static str] {
        &["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"]
    }

    /// Creates an AudioFormats enum from a file path based on its extension
    #[inline]
    pub fn from_path(value: impl AsRef<Path>) -> Option<Self> {
        Some(
            match value
                .as_ref()
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase()
                .as_ref()
            {
                "wav" => Self::Wav,
                "mp3" => Self::Mp3,
                "flac" => Self::Flac,
                "ogg" => Self::Ogg,
                "m4a" => Self::M4a,
                "aac" => Self::Aac,
                "opus" => Self::Opus,
                _ => return None,
            },
        )
    }
}

/// Configuration options for audio normalization process
#[derive(Debug, Clone)]
pub struct NormalizationOptions {
    /// Input directory containing audio files to process
    pub input_dir: PathBuf,
    /// Output directory for normalized audio files. If not set, override the audio in input dir.
    pub output_dir: Option<PathBuf>,
    /// Percentage of files to sample for calculating target loudness (0.0 to 1.0)
    pub sample_percentage: f64,
    /// Percentage of measurements to trim when calculating average loudness (0.0 to 0.5)
    pub trim_percentage: f64,
    /// Target loudness in LUFS (Loudness Units Full Scale)
    pub target_lufs: Option<f64>,
    /// Target true peak in dBTP (decibels True Peak)
    pub true_peak_db: f64,
    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
}

impl Default for NormalizationOptions {
    fn default() -> Self {
        NormalizationOptions {
            input_dir: PathBuf::from("."),
            output_dir: None,
            sample_percentage: 0.30,
            trim_percentage: 0.30,
            target_lufs: None,
            true_peak_db: -1.5,
            num_threads: None,
        }
    }
}

/// Represents an audio file to be processed
#[derive(Debug)]
struct AudioFile {
    path: PathBuf,
}

/// Normalize the loudness of all audio files in a folder
pub fn normalize_folder_loudness(options: &NormalizationOptions) -> Result<(), Error> {
    // Configure Rayon thread pool size if specified
    if let Some(num_threads) = options.num_threads {
        if num_threads > 0 {
            let rayon_init_result = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global();
            if let Err(e) = rayon_init_result {
                warn!(
                    "Failed to configure Rayon thread pool: {}. Using default number of threads.",
                    e
                );
            } else {
                info!("Using {} threads for processing.", num_threads);
            }
        } else {
            info!("Using default number of threads.");
        }
    } else {
        info!("Using default number of threads.");
    }

    let mut loudness_measurements_cache = None;

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

    // --- Target Loudness Calculation ---
    let target_lufs = match options.target_lufs {
        Some(t) => {
            info!("Using user-provided Target Loudness: {:.2} LUFS", t);
            t
        }
        None => {
            // 3. Select sample files
            let sample_size = (all_audio_files.len() as f64 * options.sample_percentage) as usize;
            if sample_size == 0 {
                return Err(Error::InvalidOptions(
                    "Sample size is 0, please check your sample percentage.".to_string(),
                ));
            }
            let sampled_files: Vec<&AudioFile> = if sample_size >= all_audio_files.len() {
                all_audio_files.iter().collect()
            } else {
                let mut rng = rand::rng();
                all_audio_files
                    .choose_multiple(&mut rng, sample_size)
                    .collect()
            };
            info!(
                "Selected {} files for target loudness calculation.",
                sampled_files.len()
            );

            // 4. Measure loudness of sampled files (in parallel)
            info!("Measuring loudness of sampled files...");
            let sample_pb = ProgressBar::new(sampled_files.len() as u64);
            sample_pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}").expect("Internal Error: Failed to set progress bar style")
                .progress_chars("#>-"));
            sample_pb.set_message("Measuring samples");

            let loudness_measurements: HashMap<PathBuf, Result<f64, MeasurementError>> =
                sampled_files
                    .par_iter()
                    .progress_with(sample_pb.clone())
                    .map(|audio_file| {
                        let loudness_result = measure_single_file_loudness(&audio_file.path);
                        if let Err(e) = &loudness_result {
                            warn!(
                                "Failed to measure loudness for sample {:?}: {}",
                                audio_file.path.file_name().unwrap_or_default(),
                                e
                            );
                        }
                        (audio_file.path.clone(), loudness_result)
                    })
                    .collect();
            sample_pb.finish_with_message("Sample measurement done");

            // 5. Calculate target loudness (trimmed mean)
            let calculated_target =
                calculate_target_loudness(&loudness_measurements, options.trim_percentage)
                    .map_err(|e| Error::Processing {
                        path: options.input_dir.to_path_buf(),
                        source: ProcessingError::TargetLoudnessCalculationFailed(e.to_string()),
                    })?;
            info!(
                "Calculated Target Loudness ({}% trimmed mean): {:.2} LUFS",
                options.trim_percentage * 100.0,
                calculated_target
            );

            loudness_measurements_cache = Some(loudness_measurements);

            calculated_target
        }
    };

    // --- Processing All Files ---
    info!(
        "Processing all {} files to target {:.2} LUFS / {:.1} dBTP...",
        all_audio_files.len(),
        target_lufs,
        options.true_peak_db
    );
    let process_pb = ProgressBar::new(all_audio_files.len() as u64);
    process_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}").expect("Internal Error: Failed to set progress bar style")
        .progress_chars("#>-"));
    process_pb.set_message("Processing files");

    let results: Vec<Result<(), Error>> = all_audio_files
        .par_iter()
        .progress_with(process_pb.clone())
        .map(|audio_file| {
            process_single_file(
                &audio_file.path,
                target_lufs,
                options.true_peak_db,
                &options.input_dir,
                &options.output_dir,
                &loudness_measurements_cache,
            )
        })
        .collect();
    process_pb.finish_with_message("Processing done");

    // 7. Report final status and errors
    let mut success_count = 0;
    let mut error_count = 0;
    for result in results {
        match result {
            Ok(_) => success_count += 1,
            Err(e) => {
                error!("Error: {}", e); // Log the detailed error
                error_count += 1;
            }
        }
    }

    info!(
        "Processing complete. {} files succeeded, {} files failed.",
        success_count, error_count
    );

    if error_count > 0 {
        Err(Error::Processing {
            path: options.input_dir.to_path_buf(),
            source: ProcessingError::FilesFailed(error_count),
        })
    } else {
        Ok(())
    }
}

/// Updates RMS (Root Mean Square) accumulators for audio data
///
/// # Arguments
/// * `planar_f32` - Audio data in planar format (separate channels)
/// * `sum_of_squares` - Accumulator for sum of squared samples
/// * `total_samples` - Counter for total number of samples processed
fn update_rms_accumulators(
    planar_f32: &[Vec<f32>],
    sum_of_squares: &mut f64,
    total_samples: &mut u64,
) {
    for channel_buffer in planar_f32 {
        for sample in channel_buffer {
            *sum_of_squares += (*sample as f64) * (*sample as f64);
        }
        // Increment total_samples by the number of samples in this channel buffer
        // Assumes all channel buffers have the same length for a given frame decode
    }
    // Add samples for the first channel only if channels exist,
    // assuming all channels have equal length per decode step
    if let Some(first_channel_buffer) = planar_f32.first() {
        *total_samples += first_channel_buffer.len() as u64;
    }
}

/// Calculates RMS (Root Mean Square) value in dBFS (decibels Full Scale)
///
/// # Arguments
/// * `sum_of_squares` - Sum of squared samples
/// * `total_samples` - Total number of samples
///
/// # Returns
/// RMS value in dBFS, or negative infinity for silence
fn calculate_rms_dbfs(sum_of_squares: f64, total_samples: u64) -> f64 {
    if total_samples == 0 {
        return f64::NEG_INFINITY; // No samples, effectively silence
    }
    let mean_square = sum_of_squares / total_samples as f64;
    if mean_square <= 0.0 {
        return f64::NEG_INFINITY; // Silence or numerical issue
    }
    let rms = mean_square.sqrt();

    // Convert RMS to dBFS (decibels relative to full scale)
    // Clamp RMS to avoid log10(0) or log10(negative)
    20.0 * rms.max(f32::EPSILON as f64).log10()
}

/// Measures the loudness of a single audio file using EBU R128 or RMS fallback
///
/// # Arguments
/// * `path` - Path to the audio file
///
/// # Returns
/// Measured loudness in LUFS (Loudness Units Full Scale) or dBFS if fallback to RMS
pub fn measure_single_file_loudness(path: impl AsRef<Path>) -> Result<f64, MeasurementError> {
    let path = path.as_ref();
    let file = fs::File::open(path).map_err(MeasurementError::Io)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(MeasurementError::Symphonia)?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or(MeasurementError::NoTrack)?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or(MeasurementError::UnsupportedFormat)?;
    let channels = track
        .codec_params
        .channels
        .ok_or(MeasurementError::UnsupportedFormat)?;
    let channel_count = channels.count();

    // Create EBU R128 state
    let mut ebu_state = EbuR128::new(channel_count as u32, sample_rate, Mode::I)
        .map_err(MeasurementError::EbuR128)?;

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(MeasurementError::Symphonia)?;

    let mut packet_count = 0;
    let mut decoded_frames_count = 0; // Use a more descriptive name

    // Accumulators for RMS fallback
    let mut rms_sum_of_squares: f64 = 0.0;
    let mut rms_total_samples: u64 = 0;

    loop {
        match format.next_packet() {
            Ok(packet) => {
                packet_count += 1;
                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        decoded_frames_count += decoded.frames() as u64; // Accumulate total frames

                        // --- Feed EBU R128 ---
                        // Convert buffer to planar f32 (handle potential errors)
                        match convert_buffer_to_planar_f32(&decoded) {
                            Ok(planar_f32) => {
                                // Feed EBU state
                                let plane_slices: Vec<&[f32]> =
                                    planar_f32.iter().map(|v| v.as_slice()).collect();
                                if let Err(e) = ebu_state.add_frames_planar_f32(&plane_slices) {
                                    // Decide if this error is fatal or skippable
                                    warn!(
                                        "EBU R128 add_frames failed for chunk in {:?}: {}. Skipping chunk for EBU.",
                                        path.file_name().unwrap_or_default(),
                                        e
                                    );
                                    // Potentially return Err(MeasurementError::EbuR128(e)) if it's critical
                                }

                                // --- Update RMS Accumulators ---
                                update_rms_accumulators(
                                    &planar_f32,
                                    &mut rms_sum_of_squares,
                                    &mut rms_total_samples,
                                );
                            }
                            Err(e) => {
                                // Failed to convert this buffer, log and skip chunk for *both* measurements
                                warn!(
                                    "Buffer conversion failed for chunk in {:?}: {}. Skipping chunk.",
                                    path.file_name().unwrap_or_default(),
                                    e
                                );
                                // Optionally return the error if conversion failure is critical
                                // return Err(e);
                            }
                        }
                    }
                    Err(SymphoniaError::DecodeError(e)) => {
                        warn!(
                            "Decode error in {:?}: {}. Skipping packet.",
                            path.file_name().unwrap_or_default(),
                            e
                        );
                        // Continue decoding if possible
                    }
                    Err(SymphoniaError::IoError(ref e))
                        if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                    {
                        debug!(
                            "Decoder reached EOF for {:?}",
                            path.file_name().unwrap_or_default()
                        );
                        break; // Expected EOF during decode
                    }
                    Err(e) => {
                        // Treat other decode errors as potentially fatal for this file
                        error!(
                            "Unhandled decoder error in {:?}: {}",
                            path.file_name().unwrap_or_default(),
                            e
                        );
                        return Err(MeasurementError::Symphonia(e));
                    }
                }
            }
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                debug!(
                    "Format reader reached EOF for {:?}",
                    path.file_name().unwrap_or_default()
                );
                break; // Expected EOF at end of stream
            }
            Err(e) => {
                error!(
                    "Error reading packet from {:?}: {}",
                    path.file_name().unwrap_or_default(),
                    e
                );
                return Err(MeasurementError::Symphonia(e));
            }
        }
    }

    debug!(
        "Finished decoding {:?}: {} packets, {} total frames decoded.",
        path.file_name().unwrap_or_default(),
        packet_count,
        decoded_frames_count // Use the accumulated frame count
    );
    // Log total samples accumulated for RMS (should be frames * channels)
    debug!(
        "Total samples accumulated for RMS calculation in {:?}: {}",
        path.file_name().unwrap_or_default(),
        rms_total_samples
    );

    // --- Attempt EBU R128 Measurement ---
    match ebu_state.loudness_global() {
        Ok(lufs) if lufs.is_finite() => {
            // EBU R128 succeeded and gave a valid number
            debug!(
                "EBU R128 measurement successful for {:?}: {:.2} LUFS",
                path.file_name().unwrap_or_default(),
                lufs
            );
            Ok(lufs)
        }
        Ok(lufs_non_finite) => {
            // EBU R128 succeeded but gave Inf or NaN
            debug!(
                "EBU R128 measurement resulted in non-finite value ({}) for {:?}. Falling back to RMS.",
                lufs_non_finite,
                path.file_name().unwrap_or_default(),
            );
            // --- Fallback to RMS ---
            let rms_dbfs = calculate_rms_dbfs(rms_sum_of_squares, rms_total_samples);
            debug!(
                // Log fallback value
                "Fallback RMS measurement for {:?}: {:.2} dBFS",
                path.file_name().unwrap_or_default(),
                rms_dbfs
            );
            Ok(rms_dbfs)
        }
        Err(e) => {
            // Other EBU R128 finalization error
            debug!(
                "EBU R128 finalization error for {:?}: {}. Cannot measure loudness. Attempting fallback to RMS.",
                path.file_name().unwrap_or_default(),
                e
            );
            let rms_dbfs = calculate_rms_dbfs(rms_sum_of_squares, rms_total_samples);
            debug!(
                // Log fallback value
                "Fallback RMS measurement for {:?}: {:.2} dBFS",
                path.file_name().unwrap_or_default(),
                rms_dbfs
            );
            Ok(rms_dbfs)
        }
    }
}

/// Calculates target loudness from a set of measurements using trimmed mean
///
/// # Arguments
/// * `measurements` - Map of file paths to their loudness measurements
/// * `trim_percentage` - Percentage of measurements to trim from each end
///
/// # Returns
/// Target loudness value in LUFS
fn calculate_target_loudness(
    measurements: &HashMap<PathBuf, Result<f64, MeasurementError>>,
    trim_percentage: f64,
) -> Result<f64, ProcessingError> {
    let valid_loudnesses: Vec<f64> = measurements
        .iter()
        .filter_map(|(_, loudness_result)| match loudness_result {
            // Filter out errors AND non-finite values like -inf (silence)
            Ok(lufs) if lufs.is_finite() => Some(*lufs),
            _ => None,
        })
        .collect();

    if valid_loudnesses.is_empty() {
        return Err(ProcessingError::TargetLoudnessCalculationFailed(
            "No valid finite loudness measurements available from the sample to calculate target."
                .to_string(),
        ));
    }

    let mut sorted_loudnesses = valid_loudnesses;
    sorted_loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = sorted_loudnesses.len();
    let trim_count_each_side = (count as f64 * trim_percentage / 2.0).floor() as usize;

    debug!(
        "Total valid samples: {}, Trimming {} from each side",
        count, trim_count_each_side
    );

    let trimmed_slice = if count > trim_count_each_side * 2 {
        &sorted_loudnesses[trim_count_each_side..count - trim_count_each_side]
    } else {
        // Not enough elements to trim, use all valid measurements
        warn!(
            "Not enough samples ({}) to perform {}% trimming. Using mean of all valid samples.",
            count,
            trim_percentage * 100.0
        );
        &sorted_loudnesses[..] // Use the whole slice
    };

    if trimmed_slice.is_empty() {
        // This should theoretically not happen if valid_loudnesses was not empty,
        // but handle defensively.
        return Err(ProcessingError::TargetLoudnessCalculationFailed(
            "Trimmed slice is empty, cannot calculate target loudness.".to_string(),
        ));
    }

    let sum: f64 = trimmed_slice.iter().sum();
    let mean = sum / trimmed_slice.len() as f64;

    // Sanity check the result
    if !mean.is_finite() {
        return Err(ProcessingError::TargetLoudnessCalculationFailed(format!(
            "Calculated target loudness is not a finite number ({:.2}). Check input sample measurements.",
            mean
        )));
    }

    Ok(mean)
}

/// Processes a single audio file: measures loudness, applies gain, and saves the result
///
/// # Arguments
/// * `input_path` - Path to input audio file
/// * `target_lufs` - Target loudness in LUFS
/// * `target_peak_db` - Target peak level in dBTP
/// * `input_base_dir` - Base directory for input files
/// * `output_base_dir` - Base directory for output files. If not set, use
/// * `cache` - Optional cache of pre-measured loudness values
pub fn process_single_file(
    input_path: impl AsRef<Path>,
    target_lufs: f64,
    target_peak_db: f64,
    input_base_dir: impl AsRef<Path>,
    output_base_dir: &Option<impl AsRef<Path>>,
    cache: &Option<HashMap<PathBuf, Result<f64, MeasurementError>>>,
) -> Result<(), Error> {
    let input_path = input_path.as_ref();
    let file_format: AudioFormats =
        AudioFormats::from_path(input_path).ok_or_else(|| Error::Processing {
            path: input_path.to_path_buf(),
            source: error::ProcessingError::UnsupportedFormat,
        })?;
    let file_name_os = input_path.file_name().unwrap_or_default();
    let file_name_str = file_name_os.to_string_lossy();
    debug!("Processing: {}", file_name_str);

    // 1. Measure current loudness
    let current_lufs = match cache.as_ref().map(|x| x.get(input_path)) {
        Some(Some(Ok(lufs_result))) => *lufs_result,
        // because MeasurementError is not cloneable, so we cannot deal with Some(Err(e))
        _ => measure_single_file_loudness(input_path).map_err(|e: MeasurementError| {
            Error::Measurement {
                path: input_path.to_path_buf(),
                source: e,
            }
        })?,
    };

    // Handle silence or measurement errors resulting in -inf
    if current_lufs.is_infinite() && current_lufs.is_sign_negative() {
        info!("Skipping processing for silent file: {}", file_name_str);
        // Optionally copy the silent file or create a silent output file
        // For now, we just skip processing it further.
        return Ok(());
    }
    if !current_lufs.is_finite() {
        warn!(
            "Skipping processing for file with non-finite loudness ({}): {}",
            current_lufs, file_name_str
        );
        return Ok(()); // Skip files that couldn't be measured properly (e.g., too short)
    }

    // 2. Calculate gain needed for target loudness
    let loudness_gain_db = target_lufs - current_lufs;
    let loudness_linear_gain = 10.0_f64.powf(loudness_gain_db / 20.0);

    // 3. Decode file, apply gain, and find peak
    let (processed_samples_interleaved, spec, initial_peak_db) =
        decode_apply_gain_measure_peak(input_path, loudness_linear_gain).map_err(|e| {
            Error::Processing {
                path: input_path.to_path_buf(),
                source: e,
            }
        })?;

    // 4. Calculate gain adjustment needed for true peak limiting
    let peak_headroom_db = target_peak_db - initial_peak_db;
    let peak_limiting_gain_db = if peak_headroom_db < 0.0 {
        // Peak exceeds target, need to reduce gain
        peak_headroom_db
    } else {
        // Peak is below target, no limiting needed
        0.0
    };
    let peak_limiting_linear_gain = 10.0_f64.powf(peak_limiting_gain_db / 20.0);

    // 5. Apply peak limiting gain (if necessary) and clamp (final safety)
    let final_linear_gain = loudness_linear_gain * peak_limiting_linear_gain;
    let target_peak_linear = 10.0_f32.powf(target_peak_db as f32 / 20.0); // Target peak as linear value

    let final_samples_interleaved: Vec<f32> = processed_samples_interleaved
        .into_iter()
        // Re-apply gain using the combined factor (more efficient than applying twice)
        // We need the original samples again, or apply the limiting gain to the already loudness-adjusted samples.
        // Let's apply limiting gain to the already adjusted samples.
        .map(|s| {
            let limited_sample = s * peak_limiting_linear_gain as f32;
            // Clamp to the linear target peak value as a final safety measure
            limited_sample.clamp(-target_peak_linear, target_peak_linear)
        })
        .collect();

    debug!(
        "  -> File: {}, Current LUFS: {:.2}, Target LUFS: {:.2}, Loudness Gain: {:.2} dB",
        file_name_str, current_lufs, target_lufs, loudness_gain_db
    );
    debug!(
        "  -> Initial Peak: {:.2} dBFS, Target Peak: {:.1} dBTP, Limiting Gain: {:.2} dB",
        initial_peak_db, target_peak_db, peak_limiting_gain_db
    );
    debug!("  -> Final Combined Linear Gain: {:.3}x", final_linear_gain);

    // 6. Determine output path
    let mut output_path = match output_base_dir.as_ref() {
        Some(obd) => {
            let relative_path =
                pathdiff::diff_paths(input_path, input_base_dir).ok_or_else(|| Error::Io {
                    path: input_path.to_path_buf(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to calculate relative path",
                    ),
                })?;
            obd.as_ref().join(relative_path)
        }
        None => input_path.to_path_buf(),
    };

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    match file_format {
        AudioFormats::Ogg => {
            output_path.set_extension(file_format.to_string());
            save_as_ogg(
                &output_path,
                spec.channels.count(),
                spec.rate,
                &final_samples_interleaved,
            )
            .map_err(|e| Error::Writing {
                path: output_path.clone(),
                source: e,
            })?;
        }
        // All other formats fallback to wav
        _ => {
            output_path.set_extension("wav");
            save_as_wav(
                &output_path,
                spec.channels.count(),
                spec.rate,
                &final_samples_interleaved,
            )
            .map_err(|e| Error::Writing {
                path: output_path.clone(),
                source: e,
            })?;
        }
    }

    debug!("Successfully wrote normalized file to {:?}", output_path);
    Ok(())
}

/// Decodes an audio file, applies gain, and measures the peak level
///
/// # Arguments
/// * `path` - Path to the audio file
/// * `linear_gain` - Linear gain factor to apply
///
/// # Returns
/// Tuple containing:
/// - Processed samples in interleaved format
/// - Signal specifications
/// - Peak level in dBFS
fn decode_apply_gain_measure_peak(
    path: impl AsRef<Path>,
    linear_gain: f64,
) -> Result<(Vec<f32>, SignalSpec, f64), ProcessingError> {
    let path = path.as_ref();
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| ProcessingError::NoTrack(path.to_path_buf()))?;

    let spec = SignalSpec::new(
        track
            .codec_params
            .sample_rate
            .ok_or(ProcessingError::MissingSampleRate)?,
        track
            .codec_params
            .channels
            .ok_or(ProcessingError::MissingChannelSpec)?,
    );

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let mut all_samples_planar: Vec<Vec<f32>> = Vec::with_capacity(spec.channels.count());
    for _ in 0..spec.channels.count() {
        all_samples_planar.push(Vec::new());
    }
    let mut max_peak_linear: f32 = 0.0;

    loop {
        match format.next_packet() {
            Ok(packet) => {
                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        // Convert buffer to planar f32
                        let current_planar = convert_buffer_to_planar_f32(&decoded)?;

                        // Append samples and find peak *after* applying gain
                        for (channel_idx, plane) in current_planar.iter().enumerate() {
                            for &sample in plane {
                                let gained_sample = sample * linear_gain as f32;
                                all_samples_planar[channel_idx].push(gained_sample);
                                // Track peak of the absolute value
                                let abs_sample = gained_sample.abs();
                                if abs_sample > max_peak_linear {
                                    max_peak_linear = abs_sample;
                                }
                            }
                        }
                    }
                    Err(SymphoniaError::DecodeError(e)) => warn!("Decode error: {}", e),
                    Err(SymphoniaError::IoError(ref e))
                        if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                    {
                        break;
                    }
                    Err(e) => return Err(ProcessingError::Symphonia(e)),
                }
            }
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(ProcessingError::Symphonia(e)),
        }
    }

    // Interleave the processed samples
    let interleaved_samples = interleave_planar_f32(&all_samples_planar);

    // Convert max linear peak to dBFS
    let max_peak_db = if max_peak_linear > 0.0 {
        20.0 * max_peak_linear.log10()
    } else {
        f32::NEG_INFINITY // Represents silence or zero signal
    };

    Ok((interleaved_samples, spec, max_peak_db as f64)) // Return peak as f64
}

/// Converts any Symphonia audio buffer to planar f32 format
///
/// # Arguments
/// * `decoded` - Reference to decoded audio buffer
///
/// # Returns
/// Vector of vectors containing audio data in planar format (one vector per channel)
fn convert_buffer_to_planar_f32(
    decoded: &AudioBufferRef<'_>,
) -> Result<Vec<Vec<f32>>, ProcessingError> {
    let num_channels = decoded.spec().channels.count();
    let mut planar_output: Vec<Vec<f32>> = Vec::with_capacity(num_channels);

    match decoded {
        AudioBufferRef::F32(buf) => {
            for plane in buf.planes().planes() {
                planar_output.push(plane.to_vec());
            }
        }
        AudioBufferRef::F64(buf) => {
            for plane in buf.planes().planes() {
                planar_output.push(plane.iter().map(|&s| s as f32).collect());
            }
        }
        AudioBufferRef::S32(buf) => {
            for plane in buf.planes().planes() {
                // 将有符号32位整数规范化到[-1.0, 1.0]范围
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| (s as f32) / (i32::MAX as f32))
                        .collect(),
                );
            }
        }
        AudioBufferRef::S24(buf) => {
            for plane in buf.planes().planes() {
                // i24的范围是-2^23到2^23-1
                let max_value = 8388607.0; // 2^23 - 1
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| {
                            let sample_value = s.inner() as f32;
                            sample_value / max_value
                        })
                        .collect(),
                );
            }
        }
        AudioBufferRef::S16(buf) => {
            for plane in buf.planes().planes() {
                // 将有符号16位整数规范化到[-1.0, 1.0]范围
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| (s as f32) / (i16::MAX as f32))
                        .collect(),
                );
            }
        }
        AudioBufferRef::U8(buf) => {
            for plane in buf.planes().planes() {
                // 将无符号8位整数规范化到[-1.0, 1.0]范围
                // u8范围是[0, 255]，需要先转换到[-128, 127]
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| ((s as i16 - 128) as f32) / 128.0)
                        .collect(),
                );
            }
        }
        _ => return Err(ProcessingError::UnsupportedFormat),
    }
    Ok(planar_output)
}

/// Interleaves planar audio samples into a single vector
///
/// # Arguments
/// * `planar_samples` - Audio data in planar format
///
/// # Returns
/// Vector containing interleaved audio samples
fn interleave_planar_f32(planar_samples: &[Vec<f32>]) -> Vec<f32> {
    if planar_samples.is_empty() {
        return Vec::new();
    }
    let num_channels = planar_samples.len();
    let num_frames = planar_samples[0].len(); // Assume all planes have the same length

    // Validate that all planes have the same length
    if planar_samples.iter().any(|p| p.len() != num_frames) {
        // This indicates a problem, maybe log a warning or return error
        // For simplicity, we'll proceed assuming they are correct, but this check is good
        warn!("Planar sample planes have different lengths during interleaving!");
    }

    let mut interleaved = Vec::with_capacity(num_frames * num_channels);

    for frame_idx in 0..num_frames {
        for plane in planar_samples {
            // Use get() for safety in case of inconsistent lengths
            if let Some(sample) = plane.get(frame_idx) {
                interleaved.push(*sample);
            } else {
                // Handle missing sample, e.g., push 0.0 or log error
                interleaved.push(0.0);
            }
        }
    }
    interleaved
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
            info!("Created output directory: {:?}", output_dir);
        } else if !output_dir.is_dir() {
            return Err(Error::InvalidOptions(format!(
                "Output path exists but is not a directory: {:?}",
                output_dir
            )));
        }
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
/// Vector of AudioFile structs representing found audio files
fn find_audio_files(input_dir: impl AsRef<Path>) -> Result<Vec<AudioFile>, Error> {
    let mut audio_files = Vec::new();
    // Define supported extensions (lowercase)

    for entry in WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok()) // Filter out directory reading errors
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if let Some(ext) = path
            .extension()
            .and_then(|os| os.to_str())
            .map(|s| s.to_lowercase())
        {
            if AudioFormats::supported_extensions().contains(&ext.as_str()) {
                audio_files.push(AudioFile {
                    path: path.to_path_buf(),
                });
            }
        }
    }
    Ok(audio_files)
}
