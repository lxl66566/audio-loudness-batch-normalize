//! Run report records, CSV/JSON serialization, and output verification.

use std::{fs, path::Path};

use indicatif::{ParallelProgressIterator, ProgressBar};
use log::{info, warn};
use pathdiff::diff_paths;
use rayon::prelude::*;
use serde_json::json;

use crate::{
    error::Error,
    measurement::{AudioStats, LoudnessMeasurements, measure_single_file_stats},
    process::{ProcessedFile, compute_gain},
    progress::progress_style,
};

/// Per-file entry of a run report.
#[derive(Debug, Clone)]
pub struct FileReport {
    /// Input file path relative to the input directory.
    pub path: String,
    /// Integrated loudness of the input; `None` when skipped.
    pub input_lufs: Option<f64>,
    /// True peak of the input; `None` when skipped.
    pub input_true_peak_db: Option<f64>,
    /// Final gain in dB, including true-peak limiting (the planned gain in
    /// measure-only mode); `None` when skipped.
    pub gain_db: Option<f64>,
    /// Whether true-peak limiting reduced the gain below the loudness gain.
    pub peak_limited: bool,
    /// True for files skipped as silent or unmeasurable.
    pub skipped: bool,
    /// Integrated loudness of the written file; present only when
    /// verification ran.
    pub output_lufs: Option<f64>,
    /// True peak of the written file; present only when verification ran.
    pub output_true_peak_db: Option<f64>,
}

/// Summary of a normalization run, written by the `report` option.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// Effective target loudness in LUFS.
    pub target_lufs: f64,
    /// Effective true-peak limit in dBTP.
    pub true_peak_db: f64,
    /// One entry per processed or skipped file.
    pub files: Vec<FileReport>,
}

/// Serialization format of a report file, selected by its extension.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ReportFormat {
    Csv,
    Json,
}

impl ReportFormat {
    pub(crate) fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_string_lossy().to_lowercase().as_str() {
            "csv" => Some(Self::Csv),
            "json" => Some(Self::Json),
            _ => None,
        }
    }
}

impl RunReport {
    const CSV_HEADER: &str = "file,input_lufs,input_true_peak_db,gain_db,peak_limited,skipped,\
                              output_lufs,output_true_peak_db";

    /// Serializes the report to CSV with a header row.
    fn to_csv(&self) -> String {
        let mut out = String::from(Self::CSV_HEADER);
        out.push('\n');
        for file in &self.files {
            let fields = [
                csv_escape(&file.path),
                csv_field(file.input_lufs),
                csv_field(file.input_true_peak_db),
                csv_field(file.gain_db),
                file.peak_limited.to_string(),
                file.skipped.to_string(),
                csv_field(file.output_lufs),
                csv_field(file.output_true_peak_db),
            ];
            out.push_str(&fields.join(","));
            out.push('\n');
        }
        out
    }

    /// Serializes the report to a JSON value.
    fn to_json(&self) -> serde_json::Value {
        json!({
            "target_lufs": self.target_lufs,
            "true_peak_db": self.true_peak_db,
            "files": self.files.iter().map(file_report_json).collect::<Vec<_>>(),
        })
    }
}

fn file_report_json(file: &FileReport) -> serde_json::Value {
    json!({
        "path": file.path,
        "input_lufs": file.input_lufs,
        "input_true_peak_db": file.input_true_peak_db,
        "gain_db": file.gain_db,
        "peak_limited": file.peak_limited,
        "skipped": file.skipped,
        "output_lufs": file.output_lufs,
        "output_true_peak_db": file.output_true_peak_db,
    })
}

/// Formats an optional measurement as a fixed-precision CSV cell.
fn csv_field(value: Option<f64>) -> String {
    value.map_or_else(String::new, |v| format!("{v:.2}"))
}

/// Quotes a CSV cell when it contains a delimiter, quote, or newline.
fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Serializes `report` to `path` in the format implied by its extension.
pub(crate) fn write_report_file(report: &RunReport, path: &Path) -> Result<(), Error> {
    let format = ReportFormat::from_path(path).ok_or_else(|| {
        Error::InvalidOptions(format!(
            "Report path must have a .csv or .json extension: {}",
            path.display()
        ))
    })?;
    let content = match format {
        ReportFormat::Csv => report.to_csv(),
        ReportFormat::Json => report.to_json().to_string(),
    };
    fs::write(path, content).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Path of `path` relative to `base_dir`, for human-readable report rows.
fn relative_display_path(path: &Path, base_dir: &Path) -> String {
    diff_paths(path, base_dir)
        .unwrap_or_else(|| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn file_report_from_processed(
    file: &ProcessedFile,
    input_base_dir: &Path,
) -> FileReport {
    FileReport {
        path: relative_display_path(&file.path, input_base_dir),
        input_lufs: file.stats.map(|s| s.loudness_lufs),
        input_true_peak_db: file.stats.map(|s| s.true_peak_db),
        gain_db: file.stats.map(|_| file.gain_db),
        peak_limited: file.peak_limited,
        skipped: file.stats.is_none(),
        output_lufs: None,
        output_true_peak_db: None,
    }
}

/// Builds a report from the sampling-phase measurements (measure-only mode).
pub(crate) fn build_measure_only_report(
    measurements: &LoudnessMeasurements,
    target_lufs: f64,
    target_peak_db: f64,
    input_base_dir: &Path,
) -> RunReport {
    let mut entries: Vec<_> = measurements.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let files = entries
        .into_iter()
        .map(|(path, result)| {
            let stats = result.as_ref().ok().filter(|s| s.loudness_lufs.is_finite());
            let (gain_db, peak_limited) = stats.map_or((None, false), |s| {
                let gain = compute_gain(*s, target_lufs, target_peak_db);
                (Some(gain.db), gain.peak_limited)
            });
            FileReport {
                path: relative_display_path(path, input_base_dir),
                input_lufs: stats.map(|s| s.loudness_lufs),
                input_true_peak_db: stats.map(|s| s.true_peak_db),
                gain_db,
                peak_limited,
                skipped: stats.is_none(),
                output_lufs: None,
                output_true_peak_db: None,
            }
        })
        .collect();
    RunReport {
        target_lufs,
        true_peak_db: target_peak_db,
        files,
    }
}

/// Re-measures written files, records the deviations in `reports`, and logs a
/// summary. Files without output (skipped) are ignored.
pub(crate) fn verify_outputs(
    pool: &rayon::ThreadPool,
    processed: &[ProcessedFile],
    reports: &mut [FileReport],
    target_lufs: f64,
    target_peak_db: f64,
) {
    info!("Verifying written files...");
    let pb = ProgressBar::new(processed.len() as u64);
    pb.set_style(progress_style());
    pb.set_message("Verifying outputs");

    let measured: Vec<Option<AudioStats>> = pool.install(|| {
        processed
            .par_iter()
            .progress_with(pb)
            .map(|file| {
                file.output_path
                    .as_deref()
                    .and_then(|path| measure_single_file_stats(path).ok())
            })
            .collect()
    });

    let mut checked = 0;
    let mut sum_abs_deviation = 0.0;
    let mut worst: Option<(String, f64)> = None;
    for (report, stats) in reports.iter_mut().zip(measured) {
        let Some(stats) = stats else { continue };
        let deviation = stats.loudness_lufs - target_lufs;
        if deviation.abs() > 1.0 {
            warn!(
                "Output of {} deviates from the target by {deviation:+.2} LU",
                report.path
            );
        }
        if stats.true_peak_db > target_peak_db + 0.5 {
            warn!(
                "Output of {} exceeds the true-peak limit: {:.2} dBTP",
                report.path, stats.true_peak_db
            );
        }
        report.output_lufs = Some(stats.loudness_lufs);
        report.output_true_peak_db = Some(stats.true_peak_db);
        checked += 1;
        sum_abs_deviation += deviation.abs();
        if worst
            .as_ref()
            .is_none_or(|(_, d)| deviation.abs() > d.abs())
        {
            worst = Some((report.path.clone(), deviation));
        }
    }
    if checked > 0 {
        let mean = sum_abs_deviation / f64::from(checked);
        if let Some((path, deviation)) = worst {
            info!(
                "Verification: {checked} files checked, mean |deviation| {mean:.2} LU, worst \
                 {deviation:+.2} LU ({path})"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FileReport, RunReport};

    #[test]
    fn report_csv_quotes_special_characters() {
        let report = RunReport {
            target_lufs: -15.0,
            true_peak_db: -1.5,
            files: vec![FileReport {
                path: "a,file,\"x\".ogg".to_string(),
                input_lufs: Some(-16.0),
                input_true_peak_db: Some(-2.0),
                gain_db: Some(1.0),
                peak_limited: false,
                skipped: false,
                output_lufs: None,
                output_true_peak_db: None,
            }],
        };
        let csv = report.to_csv();
        let rows: Vec<&str> = csv.lines().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            "file,input_lufs,input_true_peak_db,gain_db,peak_limited,skipped,output_lufs,\
             output_true_peak_db"
        );
        assert!(rows[1].contains("\"a,file,\"\"x\"\".ogg\""));
    }
}
