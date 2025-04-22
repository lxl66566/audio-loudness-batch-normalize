use anyhow::Result;
use audio_loudness_batch_normalize::{NormalizationOptions, normalize_folder_loudness};
use clap::Parser;
use log::{error, info};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// input directory
    input: PathBuf,

    /// output directory, default to override input audios
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// percentage of files to sample for calculating target loudness, default to sample all files
    #[arg(short, long, default_value_t = 1.00)]
    sample_percentage: f64,

    /// trim percentage for calculating target loudness
    #[arg(long, default_value_t = 0.30)]
    trim_percentage: f64,

    /// target loudness in LUFS, default to automatic calculation
    #[arg(long)]
    target_lufs: Option<f64>,

    /// target true peak in dBTP
    #[arg(long, default_value_t = -1.5)]
    true_peak_db: f64,

    /// number of threads to use, default to CPU core count
    #[arg(short, long)]
    threads: Option<usize>,
}

fn main() -> Result<()> {
    _ = pretty_env_logger::formatted_builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_secs()
        .parse_filters("symphonia=error")
        .parse_default_env()
        .try_init();

    let cli = Cli::parse();

    // --- Configuration ---
    let options = NormalizationOptions {
        output_dir: cli.output,
        input_dir: cli.input,
        sample_percentage: cli.sample_percentage,
        trim_percentage: cli.trim_percentage,
        target_lufs: cli.target_lufs,
        true_peak_db: cli.true_peak_db,
        num_threads: cli.threads,
    };

    info!("Starting loudness normalization with options:");
    info!("  Input Directory: {:?}", options.input_dir);
    info!("  Output Directory: {:?}", options.output_dir);
    info!(
        "  Sample Percentage: {}%",
        options.sample_percentage * 100.0
    );
    info!("  Trim Percentage: {}%", options.trim_percentage * 100.0);
    if let Some(t) = options.target_lufs {
        info!("  Target Loudness: {:.2} LUFS", t);
    } else {
        info!("  Target Loudness: Automatic (calculated)");
    }
    info!("  Target True Peak: {:.1} dBTP", options.true_peak_db);
    if let Some(n) = options.num_threads {
        info!("  Threads: {}", n);
    } else {
        info!("  Threads: Default");
    }
    info!("---");

    match normalize_folder_loudness(&options) {
        Ok(_) => {
            info!("Normalization finished successfully!");
            Ok(())
        }
        Err(e) => {
            error!("Normalization failed: {}", e);
            Err(e)?
        }
    }
}
