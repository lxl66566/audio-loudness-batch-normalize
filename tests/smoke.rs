// Test signal generation narrows float amplitudes to integer samples with `as`.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{num::NonZero, path::Path};

use audio_loudness_batch_normalize::{
    EncodeConfig, NormalizationOptions, WavBitDepth, measure_single_file_stats,
    normalize_folder_loudness, process_single_file,
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

#[test]
fn run_report_json_records_verification() {
    let dir = tempfile::tempdir().unwrap();
    // Both amplitudes stay far enough from the peak limit at the resulting
    // target for neither file to be limited.
    write_sine_wav(&dir.path().join("quiet.wav"), 0.03);
    write_sine_wav(&dir.path().join("loud.wav"), 0.1);
    let out_dir = tempfile::tempdir().unwrap();
    let report_path = dir.path().join("sub").join("report.json");

    let options = NormalizationOptions {
        input_dir: dir.path().to_path_buf(),
        output_dir: Some(out_dir.path().to_path_buf()),
        report: Some(report_path.clone()),
        verify: true,
        ..NormalizationOptions::default()
    };
    normalize_folder_loudness(&options).unwrap();

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    let files = report["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    let target = report["target_lufs"].as_f64().unwrap();
    assert!(target.is_finite());
    for file in files {
        assert_eq!(file["skipped"].as_bool(), Some(false));
        assert!(file["input_lufs"].as_f64().unwrap().is_finite());
        assert!(file["gain_db"].as_f64().unwrap().is_finite());
        assert!(!file["peak_limited"].as_bool().unwrap());
        // Verification re-measured every written file.
        let output_lufs = file["output_lufs"].as_f64().unwrap();
        assert!(
            (output_lufs - target).abs() < 0.5,
            "WAV output off target: {output_lufs} vs {target}"
        );
        assert!(file["output_true_peak_db"].as_f64().unwrap() <= -1.0);
    }
}

#[test]
fn measure_only_report_marks_skipped_files() {
    let dir = tempfile::tempdir().unwrap();
    write_sine_wav(&dir.path().join("sine.wav"), 0.3);
    write_sine_wav(&dir.path().join("silent.wav"), 0.0);
    let report_path = dir.path().join("report.csv");

    let options = NormalizationOptions {
        input_dir: dir.path().to_path_buf(),
        report: Some(report_path.clone()),
        measure_only: true,
        ..NormalizationOptions::default()
    };
    normalize_folder_loudness(&options).unwrap();

    let content = std::fs::read_to_string(&report_path).unwrap();
    let mut lines = content.lines();
    assert_eq!(
        lines.next(),
        Some(
            "file,input_lufs,input_true_peak_db,gain_db,peak_limited,skipped,output_lufs,\
             output_true_peak_db"
        )
    );
    let rows: Vec<Vec<&str>> = lines.map(|line| line.split(',').collect()).collect();
    assert_eq!(rows.len(), 2);
    let sine = rows.iter().find(|row| row[0] == "sine.wav").unwrap();
    let silent = rows.iter().find(|row| row[0] == "silent.wav").unwrap();
    assert_eq!(sine[5], "false");
    assert_ne!(sine[1], "");
    assert_ne!(sine[3], "");
    assert_eq!(silent[5], "true");
    assert_eq!(silent[1], "");
    assert_eq!(silent[3], "");
}

#[test]
fn verify_conflicts_with_measure_only() {
    let dir = tempfile::tempdir().unwrap();
    let options = NormalizationOptions {
        input_dir: dir.path().to_path_buf(),
        measure_only: true,
        verify: true,
        ..NormalizationOptions::default()
    };
    assert!(normalize_folder_loudness(&options).is_err());
}

#[test]
fn report_rejects_unknown_extension() {
    let dir = tempfile::tempdir().unwrap();
    write_sine_wav(&dir.path().join("sine.wav"), 0.3);
    let options = NormalizationOptions {
        input_dir: dir.path().to_path_buf(),
        report: Some(dir.path().join("report.txt")),
        ..NormalizationOptions::default()
    };
    assert!(normalize_folder_loudness(&options).is_err());
}
