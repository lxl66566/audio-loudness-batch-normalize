use std::{
    fs::File,
    num::NonZero,
    path::Path,
};

use log::warn;
use symphonia::core::{audio::SignalSpec, errors::Error as SymphoniaError};
use vorbis_rs::VorbisEncoderBuilder;

use crate::{
    convert_buffer_to_planar_f32,
    error::{Error, WritingError},
};

/// Helper for `decode_apply_gain_and_save`: streams processed data directly to
/// a WAV writer.
pub fn stream_to_wav_writer(
    format: &mut dyn symphonia::core::formats::FormatReader,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    final_linear_gain: f64,
    output_path: &Path,
    spec: SignalSpec,
) -> Result<(), Error> {
    let hound_spec = hound::WavSpec {
        channels: spec.channels.count() as u16,
        sample_rate: spec.rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(output_path, hound_spec).map_err(|e| Error::Writing {
            path: output_path.to_path_buf(),
            source: WritingError::Wav(e),
        })?;
    loop {
        match format.next_packet() {
            Ok(packet) => match decoder.decode(&packet) {
                Ok(decoded) => {
                    let planar =
                        convert_buffer_to_planar_f32(&decoded).map_err(|e| Error::Processing {
                            path: output_path.to_path_buf(),
                            source: e,
                        })?;

                    if planar.is_empty() || planar[0].is_empty() {
                        continue;
                    }

                    let num_frames = planar[0].len();
                    let num_channels = planar.len();

                    for frame_idx in 0..num_frames {
                        for channel_idx in 0..num_channels {
                            let sample = planar[channel_idx][frame_idx];
                            let processed_sample = sample * final_linear_gain as f32;
                            writer
                                .write_sample(processed_sample)
                                .map_err(|e| Error::Writing {
                                    path: output_path.to_path_buf(),
                                    source: WritingError::Wav(e),
                                })?;
                        }
                    }
                }
                Err(SymphoniaError::DecodeError(e)) => {
                    warn!("Decode error during final write: {e}")
                }
                Err(e) => {
                    return Err(Error::Processing {
                        path: output_path.to_path_buf(),
                        source: e.into(),
                    });
                }
            },
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(Error::Processing {
                    path: output_path.to_path_buf(),
                    source: e.into(),
                });
            }
        }
    }
    writer.finalize().map_err(|e| Error::Writing {
        path: output_path.to_path_buf(),
        source: WritingError::Wav(e),
    })
}

pub fn stream_to_ogg_writer(
    format: &mut dyn symphonia::core::formats::FormatReader,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    final_linear_gain: f64,
    output_path: &Path,
    spec: SignalSpec,
) -> Result<(), Error> {
    // OGG saving still requires collecting all samples due to vorbis-rs API.
    let samples =
        collect_gained_samples(&mut *format, &mut *decoder, final_linear_gain, output_path)?;
    save_as_ogg(output_path, spec.channels.count(), spec.rate, &samples).map_err(|e| {
        Error::Writing {
            path: output_path.to_path_buf(),
            source: e,
        }
    })?;
    Ok(())
}

/// Saves audio data as an Ogg Vorbis file
///
/// # Arguments
/// * `path` - Output file path
/// * `channels` - Number of audio channels
/// * `sample_rate` - Sample rate in Hz
/// * `samples` - Interleaved audio samples in 32-bit float format
///
/// # Returns
/// Result indicating success or a WritingError
///
/// # Panics
/// Panics if the number of channels is zero
pub fn save_as_ogg(
    path: &Path,
    channels: usize,
    sample_rate: u32,
    samples: &[f32],
) -> Result<(), WritingError> {
    assert!(channels > 0, "channels could not be zero");
    // Open the output file
    let output_file = File::create(path)?;

    // Initialize the Vorbis encoder
    let mut encoder = VorbisEncoderBuilder::new(
        NonZero::new(sample_rate).unwrap(),
        NonZero::new(channels as u8).unwrap(),
        output_file,
    )?
    .build()?;

    // Convert interleaved samples to planar format
    let mut planar_samples: Vec<Vec<f32>> = vec![Vec::new(); channels];
    for (i, &sample) in samples.iter().enumerate() {
        let channel = i % channels;
        planar_samples[channel].push(sample);
        if channel * sample_rate as usize * 2 == i + 1 {
            encoder.encode_audio_block(&planar_samples)?;
            planar_samples = vec![Vec::new(); channels];
        }
    }
    encoder.encode_audio_block(&planar_samples)?;
    encoder.finish()?;
    Ok(())
}

/// Helper for `decode_apply_gain_and_save`: collects all processed samples into
/// a Vec. Used for formats that don't support streaming writes with the current
/// libraries (e.g., Ogg).
fn collect_gained_samples(
    format: &mut dyn symphonia::core::formats::FormatReader,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    final_linear_gain: f64,
    output_path: &Path, // for error context
) -> Result<Vec<f32>, Error> {
    let mut all_samples_interleaved = Vec::new();
    loop {
        match format.next_packet() {
            Ok(packet) => match decoder.decode(&packet) {
                Ok(decoded) => {
                    let planar =
                        convert_buffer_to_planar_f32(&decoded).map_err(|e| Error::Processing {
                            path: output_path.to_path_buf(),
                            source: e,
                        })?;

                    if planar.is_empty() || planar[0].is_empty() {
                        continue;
                    }

                    let num_frames = planar[0].len();
                    let num_channels = planar.len();

                    for frame_idx in 0..num_frames {
                        for channel_idx in 0..num_channels {
                            let sample = planar[channel_idx][frame_idx];
                            let processed_sample = sample * final_linear_gain as f32;
                            all_samples_interleaved.push(processed_sample);
                        }
                    }
                }
                Err(SymphoniaError::DecodeError(e)) => {
                    warn!("Decode error during sample collection: {e}")
                }
                Err(e) => {
                    return Err(Error::Processing {
                        path: output_path.to_path_buf(),
                        source: e.into(),
                    });
                }
            },
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(Error::Processing {
                    path: output_path.to_path_buf(),
                    source: e.into(),
                });
            }
        }
    }
    Ok(all_samples_interleaved)
}
