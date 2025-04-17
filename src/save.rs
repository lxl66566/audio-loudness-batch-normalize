use vorbis_rs::VorbisEncoderBuilder;

use crate::error::WritingError;
use std::{fs::File, num::NonZero, path::Path};

pub fn save_as_wav(
    path: &Path,
    channels: usize,
    sample_rate: u32,
    samples: &[f32],
) -> Result<(), WritingError> {
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    Ok(writer.finalize()?)
}

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
