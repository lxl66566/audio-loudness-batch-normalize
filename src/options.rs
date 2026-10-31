//! Normalization options and their validation.

use std::{fs, path::PathBuf};

use log::{info, warn};

use crate::{error::Error, format::WavBitDepth, report::ReportFormat};

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
    /// sample format for WAV output files, default to matching lossless
    /// integer sources
    #[cfg_attr(
        feature = "bin",
        arg(long, value_enum, default_value_t = Self::DEFAULT_WAV_BIT_DEPTH)
    )]
    pub wav_bit_depth: WavBitDepth,
    /// Ogg Vorbis perceptual quality (0.0 to 10.0)
    #[cfg_attr(
        feature = "bin",
        arg(long, default_value_t = Self::DEFAULT_OGG_QUALITY)
    )]
    pub ogg_quality: f64,
    /// write a per-file measurement report; the format is selected by the
    /// path extension (.csv or .json)
    #[cfg_attr(feature = "bin", arg(long))]
    pub report: Option<PathBuf>,
    /// re-measure written files and record the deviations in the report
    #[cfg_attr(feature = "bin", arg(long))]
    pub verify: bool,
}

impl NormalizationOptions {
    /// Default Ogg Vorbis quality on the 0-10 scale.
    pub const DEFAULT_OGG_QUALITY: f64 = 5.0;
    /// Default percentage of files to sample for target loudness calculation.
    pub const DEFAULT_SAMPLE_PERCENTAGE: f64 = 1.0;
    /// Default percentage of measurements to trim from each end.
    pub const DEFAULT_TRIM_PERCENTAGE: f64 = 0.3;
    /// Default target true peak in dBTP.
    pub const DEFAULT_TRUE_PEAK_DB: f64 = -1.5;
    /// Default WAV output sample format.
    pub const DEFAULT_WAV_BIT_DEPTH: WavBitDepth = WavBitDepth::Auto;
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
            wav_bit_depth: Self::DEFAULT_WAV_BIT_DEPTH,
            ogg_quality: Self::DEFAULT_OGG_QUALITY,
            report: None,
            verify: false,
        }
    }
}

/// Validates normalization options for correctness.
pub(crate) fn validate_options(options: &NormalizationOptions) -> Result<(), Error> {
    if !options.input_dir.is_dir() {
        return Err(Error::InvalidOptions(format!(
            "Input path is not a valid directory: {}",
            options.input_dir.display()
        )));
    }
    if let Some(output_dir) = &options.output_dir {
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).map_err(|e| Error::Io {
                path: output_dir.clone(),
                source: e,
            })?;
            info!("Created output directory: {}", output_dir.display());
        } else if !output_dir.is_dir() {
            return Err(Error::InvalidOptions(format!(
                "Output path exists but is not a directory: {}",
                output_dir.display()
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
    if !(0.0..=10.0).contains(&options.ogg_quality) {
        return Err(Error::InvalidOptions(format!(
            "OGG quality must be between 0.0 and 10.0: {}",
            options.ogg_quality
        )));
    }
    if options.measure_only && options.verify {
        return Err(Error::InvalidOptions(
            "verify cannot be combined with measure-only".to_string(),
        ));
    }
    if let Some(report_path) = &options.report {
        if ReportFormat::from_path(report_path).is_none() {
            return Err(Error::InvalidOptions(format!(
                "Report path must have a .csv or .json extension: {}",
                report_path.display()
            )));
        }
        if let Some(parent) = report_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| Error::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
    }
    if options.true_peak_db > 0.0 {
        warn!(
            "Target true peak {:.1} dBTP is above 0 dBFS. This will likely cause clipping in \
             standard formats.",
            options.true_peak_db
        );
    }
    Ok(())
}
