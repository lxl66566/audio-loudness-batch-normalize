//! Shared progress bar styling for the parallel stages.

use indicatif::ProgressStyle;

/// Shared progress bar style for the parallel stages.
const PROGRESS_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}";

/// Builds the shared progress bar style.
pub(crate) fn progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(PROGRESS_TEMPLATE)
        .expect("internal error: invalid progress bar template")
        .progress_chars("#>-")
}
