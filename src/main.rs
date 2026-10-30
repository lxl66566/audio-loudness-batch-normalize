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
    init_logger()?;

    // --- Configuration ---
    let options = Cli::parse().options;

    info!("Starting loudness normalization with options:");
    info!("  Input Directory: {}", options.input_dir.display());
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
    info!("  WAV Bit Depth: {:?}", options.wav_bit_depth);
    info!("  OGG Quality: {}", options.ogg_quality);
    if let Some(path) = &options.report {
        info!("  Report: {}", path.display());
    }
    info!("  Verify: {}", options.verify);
    if let Some(n) = options.num_threads {
        info!("  Threads: {n}");
    } else {
        info!("  Threads: Default");
    }
    info!("  Measure Only: {}", options.measure_only);
    info!("---");

    match normalize_folder_loudness(&options) {
        Ok(()) => {
            info!("Normalization finished successfully!");
            Ok(())
        },
        Err(e) => {
            error!("Normalization failed: {e}");
            Err(e)?
        },
    }
}

/// Routes `log` crate records to spdlog. Defaults to info level with symphonia
/// at error level; `RUST_LOG` overrides the defaults.
fn init_logger() -> Result<()> {
    spdlog::init_log_crate_proxy()?;

    let mut filter = env_filter::Builder::new();
    filter
        .filter_level(log::LevelFilter::Info)
        .filter_module("symphonia", log::LevelFilter::Error);
    if let Ok(spec) = std::env::var("RUST_LOG") {
        filter.parse(&spec);
    }
    spdlog::log_crate_proxy().set_filter(Some(filter.build()));

    // Let the filter decide; the default logger accepts all levels.
    spdlog::default_logger().set_level_filter(spdlog::LevelFilter::All);
    log::set_max_level(log::LevelFilter::Trace);
    Ok(())
}
