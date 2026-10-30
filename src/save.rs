use std::{num::NonZero, path::Path};

use tempfile::NamedTempFile;
use vorbis_rs::VorbisEncoderBuilder;

use crate::{
    decode::DecodedAudio,
    error::{DecodeError, Error, ProcessingError, WritingError},
};

/// Error while saving decoded audio: a decode failure on the input, or a
/// write failure on the output.
#[derive(thiserror::Error, Debug)]
enum SaveError {
    #[error(transparent)]
    Decode(#[from] DecodeError),
    #[error(transparent)]
    Write(#[from] WritingError),
}

/// Decodes the remaining audio, applies the linear gain, and writes the
/// result as a 32-bit float WAV file.
///
/// The output is written to a temporary file in the output directory and
/// atomically renamed over `output_path` on success, so a failure (or an
/// in-place overwrite) never truncates an existing file before the new
/// content is complete.
pub fn save_wav(
    audio: &mut DecodedAudio,
    input_path: &Path,
    output_path: &Path,
    final_linear_gain: f64,
) -> Result<(), Error> {
    let temp = create_temp_file(output_path)?;
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let file = temp
        .reopen()
        .map_err(|e| write_error(output_path, e.into()))?;
    let mut writer =
        hound::WavWriter::new(file, spec).map_err(|e| write_error(output_path, e.into()))?;

    audio
        .for_each_packet(|planar| {
            for frame_idx in 0..planar[0].len() {
                for channel in planar {
                    writer
                        .write_sample(channel[frame_idx] * final_linear_gain as f32)
                        .map_err(|e| SaveError::Write(e.into()))?;
                }
            }
            Ok(())
        })
        .map_err(|e| map_save_error(e, input_path, output_path))?;

    writer
        .finalize()
        .map_err(|e| write_error(output_path, e.into()))?;
    persist(temp, output_path)
}

/// Decodes the remaining audio, applies the linear gain, and writes the
/// result as an Ogg Vorbis file, encoding each decoded packet as it arrives.
///
/// The output is written to a temporary file in the output directory and
/// atomically renamed over `output_path` on success, so a failure (or an
/// in-place overwrite) never truncates an existing file before the new
/// content is complete.
pub fn save_ogg(
    audio: &mut DecodedAudio,
    input_path: &Path,
    output_path: &Path,
    final_linear_gain: f64,
) -> Result<(), Error> {
    let sample_rate = NonZero::new(audio.sample_rate).ok_or_else(|| {
        write_error(
            output_path,
            WritingError::InvalidParams("sample rate is zero"),
        )
    })?;
    let channels = u8::try_from(audio.channels)
        .ok()
        .and_then(NonZero::new)
        .ok_or_else(|| {
            write_error(
                output_path,
                WritingError::InvalidParams("channels must be in 1..=255"),
            )
        })?;

    let temp = create_temp_file(output_path)?;
    let file = temp
        .reopen()
        .map_err(|e| write_error(output_path, e.into()))?;
    let mut encoder = VorbisEncoderBuilder::new(sample_rate, channels, file)
        .and_then(|mut builder| builder.build())
        .map_err(|e| write_error(output_path, e.into()))?;

    audio
        .for_each_packet(|planar| {
            let gained: Vec<Vec<f32>> = planar
                .iter()
                .map(|channel| {
                    channel
                        .iter()
                        .map(|&sample| sample * final_linear_gain as f32)
                        .collect()
                })
                .collect();
            encoder
                .encode_audio_block(&gained)
                .map_err(|e| SaveError::Write(e.into()))
        })
        .map_err(|e| map_save_error(e, input_path, output_path))?;

    // finish writes the pending blocks and returns the sink; close it before
    // the rename.
    drop(
        encoder
            .finish()
            .map_err(|e| write_error(output_path, e.into()))?,
    );
    persist(temp, output_path)
}

/// Creates a temporary file in the directory of `output_path`.
fn create_temp_file(output_path: &Path) -> Result<NamedTempFile, Error> {
    let parent = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    NamedTempFile::new_in(parent).map_err(|e| write_error(output_path, e.into()))
}

/// Atomically renames the temporary file over `output_path`.
fn persist(temp: NamedTempFile, output_path: &Path) -> Result<(), Error> {
    // persist returns the (still open) file handle; close it explicitly.
    let file = temp
        .persist(output_path)
        .map_err(|e| write_error(output_path, e.error.into()))?;
    drop(file);
    Ok(())
}

fn write_error(path: &Path, source: WritingError) -> Error {
    Error::Writing {
        path: path.to_path_buf(),
        source,
    }
}

fn map_save_error(e: SaveError, input_path: &Path, output_path: &Path) -> Error {
    match e {
        SaveError::Decode(e) => Error::Processing {
            path: input_path.to_path_buf(),
            source: ProcessingError::Decode(e),
        },
        SaveError::Write(e) => Error::Writing {
            path: output_path.to_path_buf(),
            source: e,
        },
    }
}
