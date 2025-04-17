use std::path::PathBuf;

use symphonia::core::errors::Error as SymphoniaError;

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
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

#[derive(thiserror::Error, Debug)]
pub enum WritingError {
    #[error("Writing wav Error: {0}")]
    Wav(#[from] hound::Error),
    #[error("Writing ogg Error: {0}")]
    Ogg(#[from] vorbis_rs::VorbisError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

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
        source: anyhow::Error, // Catch-all for symphonia/hound during processing
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
}
