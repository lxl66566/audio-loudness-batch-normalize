use std::path::PathBuf;
use symphonia::core::errors::Error as SymphoniaError;

#[warn(dead_code)]
#[derive(thiserror::Error, Debug)]
pub enum MeasurementError {
    #[error("Symphonia error: {0}")]
    Symphonia(#[from] SymphoniaError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("EBU R128 error: {0}")]
    EbuR128(#[from] ebur128::Error),
    #[error("No compatible audio track found")]
    NoTrack,
    #[error("Unsupported sample format")]
    UnsupportedFormat,
}

#[warn(dead_code)]
#[derive(thiserror::Error, Debug)]
pub enum WritingError {
    #[error("Writing wav Error: {0}")]
    Wav(#[from] hound::Error),
    #[error("Writing ogg Error: {0}")]
    Ogg(#[from] vorbis_rs::VorbisError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[warn(dead_code)]
#[derive(thiserror::Error, Debug)]
pub enum ProcessingError {
    #[error("Symphonia error: {0}")]
    Symphonia(#[from] SymphoniaError),
    #[error("No compatible audio track found in {0:?}")]
    NoTrack(PathBuf),
    #[error("Missing sample rate")]
    MissingSampleRate,
    #[error("Missing channel specification")]
    MissingChannelSpec,
    #[error("Unsupported audio buffer format")]
    UnsupportedFormat,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0} files failed during processing")]
    FilesFailed(usize),
    #[error("Target loudness calculation failed: {0}")]
    TargetLoudnessCalculationFailed(String),
    #[error("EBU R128 error: {0}")]
    EbuR128(#[from] ebur128::Error),
}

#[warn(dead_code)]
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Measurement failed: {source}")]
    Measurement {
        path: PathBuf,
        #[source]
        source: MeasurementError,
    },
    #[error("Audio decoding/processing failed for {path}: {source}")]
    Processing {
        path: PathBuf,
        #[source]
        source: ProcessingError,
    },
    #[error("Audio writing failed for {path}: {source}")]
    Writing {
        path: PathBuf,
        #[source]
        source: WritingError,
    },
    #[error("I/O error during processing of {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Invalid options: {0}")]
    InvalidOptions(String),
}
