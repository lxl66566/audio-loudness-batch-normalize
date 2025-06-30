#![allow(clippy::needless_range_loop)]

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
use strum_macros::Display;
use symphonia::core::{
    audio::{AudioBufferRef, SignalSpec},
    errors::Error as SymphoniaError,
    io::MediaSourceStream,
    probe::Hint,
};
use walkdir::WalkDir;

use crate::{
    error::{Error, MeasurementError, ProcessingError},
    save::{stream_to_ogg_writer, stream_to_wav_writer},
};

/// Represents supported audio file formats
#[derive(Debug, PartialEq, Display, Clone)]
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
    /// Output directory for normalized audio files. If not set, override the
    /// audio in input dir.
    pub output_dir: Option<PathBuf>,
    /// Percentage of files to sample for calculating target loudness (0.0 to
    /// 1.0)
    pub sample_percentage: f64,
    /// Percentage of measurements to trim when calculating average loudness
    /// (0.0 to 0.5)
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
    if let Some(num_threads) = options.num_threads {
        if num_threads > 0 {
            let rayon_init_result = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global();
            if let Err(e) = rayon_init_result {
                warn!(
                    "Failed to configure Rayon thread pool: {e}. Using default number of threads."
                );
            } else {
                info!("Using {num_threads} threads for processing.");
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
            info!("Using user-provided Target Loudness: {t:.2} LUFS");
            t
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
                        source: e,
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
    cache: &Option<HashMap<PathBuf, Result<f64, MeasurementError>>>,
) -> Result<(), Error> {
    let input_path = input_path.as_ref();
    let file_format = AudioFormats::from_path(input_path).ok_or_else(|| Error::Processing {
        path: input_path.to_path_buf(),
        source: ProcessingError::UnsupportedFormat,
    })?;
    let file_name_str = input_path.file_name().unwrap_or_default().to_string_lossy();
    debug!("Processing: {file_name_str}");

    // --- MEASUREMENT ---

    // 1. Measure current loudness (from cache or by reading the file)
    let current_lufs = match cache.as_ref().and_then(|x| x.get(input_path)) {
        Some(Ok(lufs_result)) => *lufs_result,
        _ => measure_single_file_loudness(input_path).map_err(|e| Error::Measurement {
            path: input_path.to_path_buf(),
            source: e,
        })?,
    };

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

    // 2. Calculate loudness gain
    let loudness_gain_db = target_lufs - current_lufs;
    let loudness_linear_gain = 10.0_f64.powf(loudness_gain_db / 20.0);

    // 3. Measure the true peak level *after* applying the loudness gain (without
    // storing audio)
    let (spec, initial_peak_db) = measure_peak_after_gain(input_path, loudness_linear_gain)
        .map_err(|e| Error::Processing {
            path: input_path.to_path_buf(),
            source: e,
        })?;

    // 4. Calculate peak limiting gain
    let peak_headroom_db = target_peak_db - initial_peak_db;
    let peak_limiting_gain_db = if peak_headroom_db < 0.0 {
        peak_headroom_db
    } else {
        0.0
    };
    let peak_limiting_linear_gain = 10.0_f64.powf(peak_limiting_gain_db / 20.0);

    // 5. Calculate the final, combined gain factor
    let final_linear_gain = loudness_linear_gain * peak_limiting_linear_gain;

    debug!(
        "  -> File: {file_name_str}, Current LUFS: {current_lufs:.2}, Target LUFS: {target_lufs:.2}, Loudness Gain: {loudness_gain_db:.2} dB"
    );
    debug!(
        "  -> Initial Peak: {initial_peak_db:.2} dBTP, Target Peak: {target_peak_db:.1} dBTP, Limiting Gain: {peak_limiting_gain_db:.2} dB"
    );
    debug!("  -> Final Combined Linear Gain: {final_linear_gain:.3}x");

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
    if !matches!(file_format, AudioFormats::Wav | AudioFormats::Ogg) {
        output_path.set_extension("wav");
    }

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    // Decode, apply final gain, and save in a streaming manner
    decode_apply_gain_and_save(
        input_path,
        &output_path,
        &file_format,
        spec,
        final_linear_gain,
    )?;

    debug!("Successfully wrote normalized file to {output_path:?}");
    Ok(())
}

/// Decodes an audio file, applies gain *in-memory* per chunk,
/// and measures the resulting true peak without storing the entire file.
///
/// # Returns
///
/// A tuple containing:
///
/// * `SignalSpec` - The signal specification of the decoded audio file
/// * `f64` - The true peak of the decoded audio file
fn measure_peak_after_gain(
    path: impl AsRef<Path>,
    linear_gain: f64,
) -> Result<(SignalSpec, f64), ProcessingError> {
    let path = path.as_ref();
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe().format(
        &Hint::new(),
        mss,
        &Default::default(),
        &Default::default(),
    )?;

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

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &Default::default())?;

    let channels_count = spec.channels.count();
    let mut ebu_state_true_peak = EbuR128::new(channels_count as u32, spec.rate, Mode::TRUE_PEAK)
        .map_err(ProcessingError::EbuR128)?;

    loop {
        match format.next_packet() {
            Ok(packet) => match decoder.decode(&packet) {
                Ok(decoded) => {
                    let current_planar = convert_buffer_to_planar_f32(&decoded)?;

                    // Apply gain to a temporary buffer for measurement
                    let gained_planar: Vec<Vec<f32>> = current_planar
                        .iter()
                        .map(|plane| plane.iter().map(|&s| s * linear_gain as f32).collect())
                        .collect();

                    // Feed EBU R128 for true peak measurement
                    let plane_slices: Vec<&[f32]> =
                        gained_planar.iter().map(|v| v.as_slice()).collect();
                    ebu_state_true_peak.add_frames_planar_f32(&plane_slices)?;
                }
                Err(SymphoniaError::DecodeError(e)) => {
                    warn!("Decode error during peak measurement: {e}")
                }
                Err(e) => return Err(ProcessingError::Symphonia(e)),
            },
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(ProcessingError::Symphonia(e)),
        }
    }

    // Get true peak from EBU R128 state and find the maximum across all channels
    let max_true_peak_db = (0..channels_count)
        .map(|i| ebu_state_true_peak.true_peak(i as u32))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);

    Ok((spec, max_true_peak_db))
}

/// Decodes the audio file again, applies the final calculated gain, and saves
/// the result in a streaming fashion to minimize memory usage.
fn decode_apply_gain_and_save(
    input_path: &Path,
    output_path: &Path,
    file_format: &AudioFormats,
    spec: SignalSpec,
    final_linear_gain: f64,
) -> Result<(), Error> {
    let file = fs::File::open(input_path).map_err(|e| Error::Io {
        path: input_path.to_path_buf(),
        source: e,
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut format = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &Default::default(), &Default::default())
        .map_err(|e| Error::Processing {
            path: input_path.to_path_buf(),
            source: e.into(),
        })?
        .format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| Error::Processing {
            path: input_path.to_path_buf(),
            source: ProcessingError::NoTrack(input_path.to_path_buf()),
        })?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(|e| Error::Processing {
            path: input_path.to_path_buf(),
            source: e.into(),
        })?;

    // --- Set up the writer based on format ---
    match file_format {
        AudioFormats::Ogg => {
            stream_to_ogg_writer(
                &mut *format,
                &mut *decoder,
                final_linear_gain,
                output_path,
                spec,
            )?;
        }
        // All other formats are saved as WAV in a streaming fashion.
        _ => {
            stream_to_wav_writer(
                &mut *format,
                &mut *decoder,
                final_linear_gain,
                output_path,
                spec,
            )?;
        }
    }

    Ok(())
}

/// Measures the loudness of a single audio file using EBU R128 or RMS fallback
pub fn measure_single_file_loudness(path: impl AsRef<Path>) -> Result<f64, MeasurementError> {
    let path = path.as_ref();
    let file = fs::File::open(path).map_err(MeasurementError::Io)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &Default::default(), &Default::default())
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

    let mut ebu_state = EbuR128::new(channel_count as u32, sample_rate, Mode::I)
        .map_err(MeasurementError::EbuR128)?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(MeasurementError::Symphonia)?;

    let mut rms_sum_of_squares: f64 = 0.0;
    let mut rms_total_samples: u64 = 0;

    loop {
        match format.next_packet() {
            Ok(packet) => match decoder.decode(&packet) {
                Ok(decoded) => match convert_buffer_to_planar_f32(&decoded) {
                    Ok(planar_f32) => {
                        let plane_slices: Vec<&[f32]> =
                            planar_f32.iter().map(|v| v.as_slice()).collect();
                        if let Err(e) = ebu_state.add_frames_planar_f32(&plane_slices) {
                            warn!("EBU R128 add_frames failed: {e}. Skipping chunk.");
                        }
                        update_rms_accumulators(
                            &planar_f32,
                            &mut rms_sum_of_squares,
                            &mut rms_total_samples,
                        );
                    }
                    Err(e) => warn!("Buffer conversion failed: {e}. Skipping chunk."),
                },
                Err(SymphoniaError::DecodeError(e)) => warn!("Decode error: {e}. Skipping packet."),
                Err(e) => return Err(MeasurementError::Symphonia(e)),
            },
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(MeasurementError::Symphonia(e)),
        }
    }

    match ebu_state.loudness_global() {
        Ok(lufs) if lufs.is_finite() => Ok(lufs),
        _ => {
            debug!("EBU R128 failed or gave non-finite value. Falling back to RMS.");
            Ok(calculate_rms_dbfs(rms_sum_of_squares, rms_total_samples))
        }
    }
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
    }
    if let Some(first_channel_buffer) = planar_f32.first() {
        *total_samples += first_channel_buffer.len() as u64;
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
    measurements: &HashMap<PathBuf, Result<f64, MeasurementError>>,
    trim_percentage: f64,
) -> Result<f64, ProcessingError> {
    let mut valid_loudnesses: Vec<f64> = measurements
        .values()
        .filter_map(|r| r.as_ref().ok().filter(|l| l.is_finite()))
        .copied()
        .collect();

    if valid_loudnesses.is_empty() {
        return Err(ProcessingError::TargetLoudnessCalculationFailed(
            "No valid finite loudness measurements available.".to_string(),
        ));
    }

    valid_loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
                let max_value = 8388607.0; // 2^23 - 1
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| s.inner() as f32 / max_value)
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
            && AudioFormats::supported_extensions().contains(&ext.as_str())
        {
            audio_files.push(AudioFile {
                path: path.to_path_buf(),
            });
        }
    }
    Ok(audio_files)
}
