use std::{num::NonZero, path::Path};

use audio_loudness_batch_normalize::{measure_single_file_stats, process_single_file};
use hound::{SampleFormat, WavSpec, WavWriter};
use vorbis_rs::VorbisEncoderBuilder;

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
        let sample = (amplitude * phase.sin() * i16::MAX as f32) as i16;
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
    process_single_file(
        &input,
        -14.0,
        -1.5,
        dir.path(),
        &Some(out_dir.path()),
        &None,
    )
    .unwrap();

    let output = out_dir.path().join("sine.wav");
    let reader = hound::WavReader::open(&output).unwrap();
    assert_eq!(reader.spec().sample_rate, 44_100);
    assert_eq!(reader.duration(), 44_100);

    let after = measure_single_file_stats(&output).unwrap();
    assert!((after.loudness_lufs - (-14.0)).abs() < 1.0);
}

#[test]
fn normalizes_in_place_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("sine.wav");
    write_sine_wav(&input, 0.03);

    process_single_file(&input, -14.0, -1.5, dir.path(), &None::<&Path>, &None).unwrap();

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

    process_single_file(&input, -14.0, -1.5, dir.path(), &None::<&Path>, &None).unwrap();

    // OGG stays OGG when no output directory is given.
    assert!(input.is_file());
    // The lossy codec moves the result slightly off the target.
    let after = measure_single_file_stats(&input).unwrap();
    assert!((after.loudness_lufs - (-14.0)).abs() < 1.5);
}
