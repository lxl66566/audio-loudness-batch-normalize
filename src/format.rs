//! Audio file formats and output sample formats.

use std::path::Path;

use strum::IntoEnumIterator as _;
use strum_macros::{Display, EnumIter};

use crate::decode::SourceSampleFormat;

/// Represents a supported audio file format. The strum serialization of each
/// variant is the corresponding file extension.
#[derive(Debug, PartialEq, Display, Clone, EnumIter)]
#[strum(serialize_all = "camelCase")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    M4a,
    Aac,
    Opus,
}

impl AudioFormat {
    /// Creates an AudioFormat from a file path based on its extension
    /// (case-insensitive).
    #[inline]
    pub fn from_path(value: impl AsRef<Path>) -> Option<Self> {
        let ext = value.as_ref().extension()?.to_string_lossy().to_lowercase();
        Self::iter().find(|format| format.to_string() == ext)
    }
}

/// Sample format for WAV output files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "bin", derive(clap::ValueEnum))]
pub enum WavBitDepth {
    /// Match the source when it is lossless integer audio (8/16-bit sources
    /// become 16-bit); float and lossy sources produce 32-bit float.
    #[default]
    Auto,
    /// 16-bit signed integer PCM.
    S16,
    /// 24-bit signed integer PCM.
    S24,
    /// 32-bit signed integer PCM.
    S32,
    /// 32-bit float.
    F32,
}

impl WavBitDepth {
    /// Resolves `Auto` against the source sample format; the result is a
    /// concrete format.
    #[must_use]
    pub fn resolve(self, source: SourceSampleFormat) -> Self {
        match (self, source) {
            (Self::Auto, SourceSampleFormat::Int8 | SourceSampleFormat::Int16) => Self::S16,
            (Self::Auto, SourceSampleFormat::Int24) => Self::S24,
            (Self::Auto, SourceSampleFormat::Int32) => Self::S32,
            (Self::Auto, SourceSampleFormat::Float | SourceSampleFormat::Unknown) => Self::F32,
            (concrete, _) => concrete,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioFormat, WavBitDepth};
    use crate::decode::SourceSampleFormat;

    #[test]
    fn format_from_path_is_case_insensitive() {
        assert_eq!(AudioFormat::from_path("a/b/c.WAV"), Some(AudioFormat::Wav));
        assert_eq!(AudioFormat::from_path("x.Mp3"), Some(AudioFormat::Mp3));
        assert_eq!(AudioFormat::from_path("y.ogg"), Some(AudioFormat::Ogg));
    }

    #[test]
    fn format_from_path_rejects_unknown_extensions() {
        assert_eq!(AudioFormat::from_path("x.txt"), None);
        assert_eq!(AudioFormat::from_path("noext"), None);
        assert_eq!(AudioFormat::from_path(".hidden"), None);
    }

    #[test]
    fn wav_bit_depth_auto_matches_lossless_integer_sources() {
        let cases = [
            (SourceSampleFormat::Int8, WavBitDepth::S16),
            (SourceSampleFormat::Int16, WavBitDepth::S16),
            (SourceSampleFormat::Int24, WavBitDepth::S24),
            (SourceSampleFormat::Int32, WavBitDepth::S32),
            (SourceSampleFormat::Float, WavBitDepth::F32),
            (SourceSampleFormat::Unknown, WavBitDepth::F32),
        ];
        for (source, expected) in cases {
            assert_eq!(WavBitDepth::Auto.resolve(source), expected);
        }
        // Explicit formats are never overridden.
        for depth in [
            WavBitDepth::S16,
            WavBitDepth::S24,
            WavBitDepth::S32,
            WavBitDepth::F32,
        ] {
            assert_eq!(depth.resolve(SourceSampleFormat::Int24), depth);
        }
    }
}
