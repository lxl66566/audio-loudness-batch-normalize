use anyhow::Result;
use audio_loudness_batch_normalize::{NormalizationOptions, normalize_folder_loudness};
use clap::Parser;
use log::{error, info};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(flatten)]
    options: NormalizationOptions,
}

fn main() -> Result<()> {
    _ = pretty_env_logger::formatted_builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_secs()
        .parse_filters("symphonia=error")
        .parse_default_env()
        .try_init();

    // --- Configuration ---
    let options = Cli::parse().options;

    info!("Starting loudness normalization with options:");
    info!("  Input Directory: {:?}", options.input_dir);
    info!("  Output Directory: {:?}", options.output_dir);
    info!(
        "  Sample Percentage: {}%",
        options.sample_percentage * 100.0
    );
    info!("  Trim Percentage: {}%", options.trim_percentage * 100.0);
    if let Some(t) = options.target_lufs {
        info!("  Target Loudness: {t:.2} LUFS");
    } else {
        info!("  Target Loudness: Automatic (calculated)");
    }
    info!("  Target True Peak: {:.1} dBTP", options.true_peak_db);
    if let Some(n) = options.num_threads {
        info!("  Threads: {n}");
    } else {
        info!("  Threads: Default");
    }
    info!("  Measure Only: {}", options.measure_only);
    info!("---");

    match normalize_folder_loudness(&options) {
        Ok(_) => {
            info!("Normalization finished successfully!");
            Ok(())
        }
        Err(e) => {
            error!("Normalization failed: {e}");
            Err(e)?
        }
    }
}
