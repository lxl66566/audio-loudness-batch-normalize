//! Batch audio loudness normalization using EBU R128 with an RMS fallback.

// Audio sample conversion and loudness statistics use `as` casts between
// integer and float domains by design; the magnitudes involved make them safe.
#![allow(clippy::needless_range_loop)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// Module for opening and decoding audio files
pub mod decode;
/// Module for error handling
pub mod error;
/// Module for audio file and output sample formats
pub mod format;
/// Module for loudness and true-peak measurement
pub mod measurement;
/// Module for normalization options and their validation
pub mod options;
/// Module for the end-to-end normalization pipeline
pub mod pipeline;
/// Module for per-file processing (gain decision, decode, save)
pub mod process;
mod progress;
/// Module for run reports and output verification
pub mod report;
/// Module for saving audio files
pub mod save;

pub use format::{AudioFormat, WavBitDepth};
pub use measurement::{
    AudioStats, LoudnessMeasurements, measure_single_file_loudness, measure_single_file_stats,
};
pub use options::NormalizationOptions;
pub use pipeline::normalize_folder_loudness;
pub use process::{ProcessedFile, process_single_file};
pub use report::{FileReport, RunReport};
pub use save::EncodeConfig;
