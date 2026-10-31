//! End-to-end normalization pipeline: option validation, file discovery,
//! target loudness calculation, parallel processing, and reporting.

use std::path::{Path, PathBuf};

use indicatif::{ParallelProgressIterator, ProgressBar};
use log::{error, info, warn};
use rand::seq::IndexedRandom as _;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::{
    error::{Error, ProcessingError},
    format::AudioFormat,
    measurement::{LoudnessMeasurements, calculate_target_loudness, measure_single_file_stats},
    options::{NormalizationOptions, validate_options},
    process::{ProcessedFile, process_single_file},
    progress::progress_style,
    report::{
        FileReport, RunReport, build_measure_only_report, file_report_from_processed,
        verify_outputs, write_report_file,
    },
    save::EncodeConfig,
};

/// Normalize the loudness of all audio files in a folder
pub fn normalize_folder_loudness(options: &NormalizationOptions) -> Result<(), Error> {
    // 1. Validate options
    validate_options(options)?;

    // 2. Discover audio files
    info!(
        "Discovering audio files in {}...",
        options.input_dir.display()
    );
    let all_audio_files = find_audio_files(&options.input_dir);
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
    let (target_lufs, loudness_measurements_cache) = if let Some(t) = options.target_lufs {
        info!("Using user-provided Target Loudness: {t:.2} LUFS");
        // Sampling is only needed without a user target, or to fill a
        // measure-only report.
        let cache = if options.measure_only && options.report.is_some() {
            Some(measure_sampled_files(&pool, &all_audio_files))
        } else {
            None
        };
        (t, cache)
    } else {
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
        let loudness_measurements = measure_sampled_files(&pool, &sampled_files);

        // 5. Calculate target loudness (trimmed mean)
        let calculated_target =
            calculate_target_loudness(&loudness_measurements, options.trim_percentage).map_err(
                |e| Error::Processing {
                    path: options.input_dir.clone(),
                    source: e,
                },
            )?;
        info!(
            "Calculated Target Loudness ({}% trimmed mean): {:.2} LUFS",
            options.trim_percentage * 100.0,
            calculated_target
        );

        (calculated_target, Some(loudness_measurements))
    };

    if options.measure_only {
        info!(
            "Measure-only mode: target loudness is {target_lufs:.2} LUFS. Skipping normalization."
        );
        if let Some(report_path) = &options.report {
            let cache = loudness_measurements_cache.unwrap_or_default();
            let report = build_measure_only_report(
                &cache,
                target_lufs,
                options.true_peak_db,
                &options.input_dir,
            );
            write_report_file(&report, report_path)?;
            info!("Report written to {}", report_path.display());
        }
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

    let encode_config = EncodeConfig {
        wav_bit_depth: options.wav_bit_depth,
        ogg_quality: options.ogg_quality,
    };
    let results: Vec<Result<ProcessedFile, Error>> = pool.install(|| {
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
                    &encode_config,
                )
            })
            .collect()
    });
    process_pb.finish_with_message("Processing done");

    // 7. Collect outcomes, verify written files, and report final status
    let mut processed: Vec<ProcessedFile> = Vec::with_capacity(results.len());
    let mut error_count = 0;
    for result in results {
        match result {
            Ok(processed_file) => processed.push(processed_file),
            Err(e) => {
                error!("Error: {e}"); // Log the detailed error
                error_count += 1;
            },
        }
    }

    info!(
        "Processing complete. {} files succeeded, {error_count} files failed.",
        processed.len()
    );

    let mut file_reports: Vec<FileReport> = processed
        .iter()
        .map(|file| file_report_from_processed(file, &options.input_dir))
        .collect();
    if options.verify {
        verify_outputs(
            &pool,
            &processed,
            &mut file_reports,
            target_lufs,
            options.true_peak_db,
        );
    }
    if let Some(report_path) = &options.report {
        let report = RunReport {
            target_lufs,
            true_peak_db: options.true_peak_db,
            files: file_reports,
        };
        write_report_file(&report, report_path)?;
        info!("Report written to {}", report_path.display());
    }

    if error_count > 0 {
        Err(Error::Processing {
            path: options.input_dir.clone(),
            source: ProcessingError::FilesFailed(error_count),
        })
    } else {
        Ok(())
    }
}

/// Measures the given files in parallel, warning about failures.
fn measure_sampled_files<I>(pool: &rayon::ThreadPool, files: I) -> LoudnessMeasurements
where
    I: IntoParallelIterator,
    I::Item: AsRef<Path>,
    I::Iter: IndexedParallelIterator,
{
    let iter = files.into_par_iter();
    let sample_pb = ProgressBar::new(iter.len() as u64);
    sample_pb.set_style(progress_style());
    sample_pb.set_message("Measuring samples");

    let measurements: LoudnessMeasurements = pool.install(|| {
        iter.progress_with(sample_pb.clone())
            .map(|path| {
                let path = path.as_ref();
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
    measurements
}

/// Finds all supported audio files in the specified directory recursively.
fn find_audio_files(input_dir: impl AsRef<Path>) -> Vec<PathBuf> {
    let mut audio_files = Vec::new();

    for entry in WalkDir::new(input_dir)
        .into_iter()
        .filter_map(Result::ok) // Filter out directory reading errors
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if AudioFormat::from_path(path).is_some() {
            audio_files.push(path.to_path_buf());
        }
    }
    audio_files
}
