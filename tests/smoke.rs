// Test signal generation narrows float amplitudes to integer samples with `as`.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{num::NonZero, path::Path};

use audio_loudness_batch_normalize::{
    EncodeConfig, WavBitDepth, measure_single_file_stats, process_single_file,
};
use hound::{SampleFormat, WavSpec, WavWriter};
use vorbis_rs::VorbisEncoderBuilder;

fn encode(ogg_quality: f64, wav_bit_depth: WavBitDepth) -> EncodeConfig {
    EncodeConfig {
        wav_bit_depth,
        ogg_quality,
    }
}

/// Writes a 1-second 440 Hz mono sine wave as a 16-bit WAV file.
fn write_sine_wav(path: &Path, amplitude: f32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for t in 0..44_100 {
        let phase = 2.0 * std::f32::consts::PI * 440.0 * t as f32 / 44_100.0;
        let sample = (amplitude * phase.sin() * f32::from(i16::MAX)) as i16;
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

/// Writes a 1-second 440 Hz mono sine wave as an Ogg Vorbis file.
fn write_sine_ogg(path: &Path, amplitude: f32) {
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = VorbisEncoderBuilder::new(
        NonZero::new(44_100).unwrap(),
        NonZero::new(1).unwrap(),
        file,
    )
    .unwrap()
    .build()
    .unwrap();
    for chunk in 0..44_100 / 1024 {
        let block: Vec<Vec<f32>> = vec![
            (0..1024)
                .map(|i| {
                    let t = chunk * 1024 + i;
                    amplitude * (2.0 * std::f32::consts::PI * 440.0 * t as f32 / 44_100.0).sin()
                })
                .collect(),
        ];
        encoder.encode_audio_block(&block).unwrap();
    }
    drop(encoder.finish().unwrap());
}

#[test]
fn measures_and_normalizes_sine_wav() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("sine.wav");
    write_sine_wav(&input, 0.3);

    let before = measure_single_file_stats(&input).unwrap();
    assert!(before.loudness_lufs.is_finite());
    assert!((-40.0..0.0).contains(&before.loudness_lufs));

    let out_dir = tempfile::tempdir().unwrap();
    let processed = process_single_file(
        &input,
        -14.0,
        -1.5,
        dir.path(),
        &Some(out_dir.path()),
        &None,
        &encode(5.0, WavBitDepth::Auto),
    )
    .unwrap();
    assert!(!processed.peak_limited);
    assert!(processed.output_path.is_some());

    let output = out_dir.path().join("sine.wav");
    let reader = hound::WavReader::open(&output).unwrap();
    assert_eq!(reader.spec().sample_rate, 44_100);
    assert_eq!(reader.duration(), 44_100);
    // Auto matches the 16-bit integer source.
    assert_eq!(reader.spec().bits_per_sample, 16);
    assert_eq!(reader.spec().sample_format, SampleFormat::Int);

    let after = measure_single_file_stats(&output).unwrap();
    assert!((after.loudness_lufs - (-14.0)).abs() < 1.0);
}

#[test]
fn wav_bit_depth_can_be_forced_to_f32() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("sine.wav");
    write_sine_wav(&input, 0.3);
    let out_dir = tempfile::tempdir().unwrap();

    process_single_file(
        &input,
        -14.0,
        -1.5,
        dir.path(),
        &Some(out_dir.path()),
        &None,
        &encode(5.0, WavBitDepth::F32),
    )
    .unwrap();

    let reader = hound::WavReader::open(out_dir.path().join("sine.wav")).unwrap();
    assert_eq!(reader.spec().bits_per_sample, 32);
    assert_eq!(reader.spec().sample_format, SampleFormat::Float);
}

#[test]
fn normalizes_in_place_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("sine.wav");
    write_sine_wav(&input, 0.03);

    process_single_file(
        &input,
        -14.0,
        -1.5,
        dir.path(),
        &None::<&Path>,
        &None,
        &encode(5.0, WavBitDepth::Auto),
    )
    .unwrap();

    // The overwritten file must still be a complete, readable WAV.
    let reader = hound::WavReader::open(&input).unwrap();
    assert_eq!(reader.spec().sample_rate, 44_100);
    assert_eq!(reader.duration(), 44_100);

    let after = measure_single_file_stats(&input).unwrap();
    assert!((after.loudness_lufs - (-14.0)).abs() < 1.0);
}

#[test]
fn normalizes_ogg_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("sine.ogg");
    write_sine_ogg(&input, 0.03);

    process_single_file(
        &input,
        -14.0,
        -1.5,
        dir.path(),
        &None::<&Path>,
        &None,
        &encode(5.0, WavBitDepth::Auto),
    )
    .unwrap();

    // OGG stays OGG when no output directory is given.
    assert!(input.is_file());
    // The lossy codec moves the result slightly off the target.
    let after = measure_single_file_stats(&input).unwrap();
    assert!((after.loudness_lufs - (-14.0)).abs() < 1.5);
}

#[test]
fn ogg_quality_controls_output_size() {
    let dir = tempfile::tempdir().unwrap();
    let low = dir.path().join("low.ogg");
    let high = dir.path().join("high.ogg");
    write_sine_ogg(&low, 0.03);
    std::fs::copy(&low, &high).unwrap();

    process_single_file(
        &low,
        -14.0,
        -1.5,
        dir.path(),
        &None::<&Path>,
        &None,
        &encode(1.0, WavBitDepth::Auto),
    )
    .unwrap();
    process_single_file(
        &high,
        -14.0,
        -1.5,
        dir.path(),
        &None::<&Path>,
        &None,
        &encode(9.0, WavBitDepth::Auto),
    )
    .unwrap();

    let size_low = std::fs::metadata(&low).unwrap().len();
    let size_high = std::fs::metadata(&high).unwrap().len();
    assert!(
        size_high > size_low,
        "quality 9 ({size_high} bytes) should produce a larger file than quality 1 ({size_low} \
         bytes)"
    );
}
