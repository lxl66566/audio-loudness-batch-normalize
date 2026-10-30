use std::{num::NonZero, path::Path};

use tempfile::NamedTempFile;
use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoderBuilder};

use crate::{
    WavBitDepth,
    decode::{DecodedAudio, MAX_S24},
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
/// result as a WAV file in the resolved sample format.
///
/// The output is written to a temporary file in the output directory and
/// atomically renamed over `output_path` on success, so a failure (or an
/// in-place overwrite) never truncates an existing file before the new
/// content is complete.
pub fn save_wav(
    mut audio: DecodedAudio,
    input_path: &Path,
    output_path: &Path,
    final_linear_gain: f64,
    bit_depth: WavBitDepth,
) -> Result<(), Error> {
    let temp = create_temp_file(output_path)?;
    let file = temp
        .reopen()
        .map_err(|e| write_error(output_path, e.into()))?;
    let depth = bit_depth.resolve(audio.source_format);
    let mut writer = hound::WavWriter::new(file, wav_spec(&audio, depth))
        .map_err(|e| write_error(output_path, e.into()))?;

    let result = match depth {
        WavBitDepth::F32 => audio.for_each_packet(|planar| {
            for frame_idx in 0..planar[0].len() {
                for channel in planar {
                    writer
                        .write_sample(channel[frame_idx] * final_linear_gain as f32)
                        .map_err(|e| SaveError::Write(e.into()))?;
                }
            }
            Ok(())
        }),
        WavBitDepth::S16 => audio.for_each_packet(|planar| {
            for frame_idx in 0..planar[0].len() {
                for channel in planar {
                    writer
                        .write_sample(quantize(
                            channel[frame_idx],
                            final_linear_gain,
                            f64::from(i16::MAX),
                        ) as i16)
                        .map_err(|e| SaveError::Write(e.into()))?;
                }
            }
            Ok(())
        }),
        WavBitDepth::S24 | WavBitDepth::S32 => {
            let full_scale = match depth {
                WavBitDepth::S24 => f64::from(MAX_S24),
                _ => f64::from(i32::MAX),
            };
            audio.for_each_packet(|planar| {
                for frame_idx in 0..planar[0].len() {
                    for channel in planar {
                        writer
                            .write_sample(quantize(
                                channel[frame_idx],
                                final_linear_gain,
                                full_scale,
                            ))
                            .map_err(|e| SaveError::Write(e.into()))?;
                    }
                }
                Ok(())
            })
        },
        WavBitDepth::Auto => unreachable!("bit depth is resolved before writing"),
    };
    result.map_err(|e| map_save_error(e, input_path, output_path))?;

    writer
        .finalize()
        .map_err(|e| write_error(output_path, e.into()))?;
    // Close the input file before renaming over it: Windows cannot replace a
    // file that is still open.
    drop(audio);
    persist(temp, output_path)
}

/// Builds the hound spec for the resolved output sample format.
fn wav_spec(audio: &DecodedAudio, depth: WavBitDepth) -> hound::WavSpec {
    let (bits_per_sample, sample_format) = match depth {
        WavBitDepth::S16 => (16, hound::SampleFormat::Int),
        WavBitDepth::S24 => (24, hound::SampleFormat::Int),
        WavBitDepth::S32 => (32, hound::SampleFormat::Int),
        WavBitDepth::F32 | WavBitDepth::Auto => (32, hound::SampleFormat::Float),
    };
    hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample,
        sample_format,
    }
}

/// Scales a sample to the target integer depth with round-to-nearest, clamped
/// to the representable range. No dithering, to keep the output deterministic
/// across runs.
fn quantize(sample: f32, gain: f64, full_scale: f64) -> i32 {
    let scaled = f64::from(sample) * gain * full_scale;
    scaled.round().clamp(-full_scale - 1.0, full_scale) as i32
}

/// Decodes the remaining audio, applies the linear gain, and writes the
/// result as an Ogg Vorbis file, encoding each decoded packet as it arrives.
/// `ogg_quality` is the perceptual quality on the 0-10 scale.
///
/// The output is written to a temporary file in the output directory and
/// atomically renamed over `output_path` on success, so a failure (or an
/// in-place overwrite) never truncates an existing file before the new
/// content is complete.
pub fn save_ogg(
    mut audio: DecodedAudio,
    input_path: &Path,
    output_path: &Path,
    final_linear_gain: f64,
    ogg_quality: f64,
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
        .and_then(|mut builder| {
            builder.bitrate_management_strategy(VorbisBitrateManagementStrategy::QualityVbr {
                target_quality: (ogg_quality / 10.0) as f32,
            });
            builder.build()
        })
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

    // Close the input file before renaming over it: Windows cannot replace a
    // file that is still open.
    drop(audio);
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

#[cfg(test)]
mod tests {
    use super::quantize;

    #[test]
    fn quantize_rounds_to_nearest() {
        assert_eq!(quantize(0.0, 1.0, f64::from(i16::MAX)), 0);
        assert_eq!(quantize(0.5, 1.0, f64::from(i16::MAX)), 16_384);
        assert_eq!(quantize(-1.0, 1.0, f64::from(i16::MAX)), -32_767);
    }

    #[test]
    fn quantize_clamps_to_representable_range() {
        assert_eq!(quantize(2.0, 1.0, f64::from(i16::MAX)), i32::from(i16::MAX));
        assert_eq!(
            quantize(-2.0, 1.0, f64::from(i16::MAX)),
            i32::from(i16::MIN)
        );
        assert_eq!(
            quantize(2.0, 1.0, f64::from(crate::decode::MAX_S24)),
            8_388_607
        );
    }
}
