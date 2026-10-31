use std::{fs, io, path::Path};

use log::warn;
use symphonia::core::{
    audio::{Audio, GenericAudioBufferRef},
    codecs::{
        CodecParameters,
        audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions},
    },
    errors::Error as SymphoniaError,
    formats::{FormatOptions, FormatReader, TrackType, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

use crate::error::DecodeError;

/// Sample format of the source audio as declared by the container/codec, used
/// to pick a matching output sample format. Lossy codecs report `Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSampleFormat {
    Int8,
    Int16,
    Int24,
    Int32,
    Float,
    Unknown,
}

impl SourceSampleFormat {
    /// Derives the source format from the codec parameters, preferring the
    /// declared sample format and falling back to the declared bit depth.
    #[must_use]
    pub fn from_codec_params(params: &AudioCodecParameters) -> Self {
        use symphonia::core::audio::sample::SampleFormat;
        if let Some(format) = params.sample_format {
            return match format {
                SampleFormat::U8 | SampleFormat::S8 => Self::Int8,
                SampleFormat::S16 => Self::Int16,
                SampleFormat::S24 => Self::Int24,
                SampleFormat::S32 => Self::Int32,
                SampleFormat::F32 | SampleFormat::F64 => Self::Float,
                _ => Self::Unknown,
            };
        }
        match params.bits_per_sample {
            Some(8) => Self::Int8,
            Some(16) => Self::Int16,
            Some(24) => Self::Int24,
            _ => Self::Unknown,
        }
    }
}

/// An opened audio file with a decoder prepared for its default audio track.
pub struct DecodedAudio {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    /// Sample rate in Hz of the default audio track.
    pub sample_rate: u32,
    /// Number of channels of the default audio track.
    pub channels: u16,
    /// Sample format of the source audio.
    pub source_format: SourceSampleFormat,
}

/// Opens an audio file and prepares a decoder for its default audio track.
pub fn open(path: &Path) -> Result<DecodedAudio, DecodeError> {
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    // The extension hint improves detection for files without magic numbers.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }

    let format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| DecodeError::NoTrack(path.to_path_buf()))?;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(CodecParameters::audio)
        .ok_or_else(|| DecodeError::NoTrack(path.to_path_buf()))?;
    let sample_rate = audio_params
        .sample_rate
        .ok_or(DecodeError::MissingSampleRate)?;
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count() as u16)
        .ok_or(DecodeError::MissingChannelSpec)?;
    let track_id = track.id;
    let source_format = SourceSampleFormat::from_codec_params(audio_params);
    let decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())?;

    Ok(DecodedAudio {
        format,
        decoder,
        track_id,
        sample_rate,
        channels,
        source_format,
    })
}

impl DecodedAudio {
    /// Decodes the remaining packets and invokes `on_packet` with the planar
    /// f32 samples of each decoded packet.
    ///
    /// Packets from other tracks are skipped. Recoverable decode errors and
    /// empty packets are skipped with a warning. A stream ending in an
    /// unexpected EOF is treated as the end of the audio.
    pub fn for_each_packet<E>(
        &mut self,
        mut on_packet: impl FnMut(&[Vec<f32>]) -> Result<(), E>,
    ) -> Result<(), E>
    where
        E: From<DecodeError>,
    {
        loop {
            let packet = match self.format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(SymphoniaError::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    break;
                },
                Err(e) => return Err(DecodeError::from(e).into()),
            };
            if packet.track_id != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(e)) => {
                    warn!("Decode error: {e}. Skipping packet.");
                    continue;
                },
                Err(e) => return Err(DecodeError::from(e).into()),
            };
            let planar = convert_buffer_to_planar_f32(&decoded)?;
            if planar.is_empty() || planar[0].is_empty() {
                continue;
            }
            on_packet(&planar)?;
        }
        Ok(())
    }
}

/// Maximum absolute value of a signed 24-bit sample: 2^23 - 1.
pub(crate) const MAX_S24: f32 = 8_388_607.0;

/// Converts a decoded audio buffer to planar f32 samples normalized to
/// [-1.0, 1.0].
pub(crate) fn convert_buffer_to_planar_f32(
    decoded: &GenericAudioBufferRef<'_>,
) -> Result<Vec<Vec<f32>>, DecodeError> {
    let num_channels = decoded.spec().channels().count();
    let mut planar_output: Vec<Vec<f32>> = Vec::with_capacity(num_channels);

    match decoded {
        GenericAudioBufferRef::F32(buf) => {
            for plane in buf.iter_planes() {
                planar_output.push(plane.to_vec());
            }
        },
        GenericAudioBufferRef::F64(buf) => {
            for plane in buf.iter_planes() {
                planar_output.push(plane.iter().map(|&s| s as f32).collect());
            }
        },
        GenericAudioBufferRef::S32(buf) => {
            for plane in buf.iter_planes() {
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| (s as f32) / (i32::MAX as f32))
                        .collect(),
                );
            }
        },
        GenericAudioBufferRef::S24(buf) => {
            for plane in buf.iter_planes() {
                planar_output.push(plane.iter().map(|&s| s.inner() as f32 / MAX_S24).collect());
            }
        },
        GenericAudioBufferRef::S16(buf) => {
            for plane in buf.iter_planes() {
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| f32::from(s) / f32::from(i16::MAX))
                        .collect(),
                );
            }
        },
        GenericAudioBufferRef::U8(buf) => {
            for plane in buf.iter_planes() {
                // Shift the unsigned range [0, 255] to signed [-128, 127]
                // first.
                planar_output.push(
                    plane
                        .iter()
                        .map(|&s| f32::from(i16::from(s) - 128) / 128.0)
                        .collect(),
                );
            }
        },
        _ => return Err(DecodeError::UnsupportedBufferFormat),
    }
    Ok(planar_output)
}

#[cfg(test)]
mod tests {
    use symphonia::core::audio::{
        AudioBuffer, AudioMut, AudioSpec, Channels, GenericAudioBufferRef, sample::i24,
    };

    use super::convert_buffer_to_planar_f32;

    #[test]
    fn converts_u8_to_planar_f32() {
        let mut buf = AudioBuffer::<u8>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 3);
        buf.resize_uninit(3);
        buf.plane_mut(0).unwrap().copy_from_slice(&[128, 255, 0]);
        let planar = convert_buffer_to_planar_f32(&GenericAudioBufferRef::U8(&buf)).unwrap();
        assert_eq!(planar.len(), 1);
        assert!((planar[0][0] - 0.0).abs() < 1e-6);
        assert!((planar[0][1] - 127.0 / 128.0).abs() < 1e-6);
        assert!((planar[0][2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn converts_s16_to_planar_f32() {
        let mut buf = AudioBuffer::<i16>::new(AudioSpec::new(44_100, Channels::Discrete(2)), 2);
        buf.resize_uninit(2);
        buf.plane_mut(0).unwrap().copy_from_slice(&[0, i16::MAX]);
        buf.plane_mut(1).unwrap().copy_from_slice(&[i16::MIN, -1]);
        let planar = convert_buffer_to_planar_f32(&GenericAudioBufferRef::S16(&buf)).unwrap();
        assert_eq!(planar.len(), 2);
        assert_eq!(planar[0][0], 0.0);
        assert_eq!(planar[0][1], 1.0);
        assert_eq!(planar[1][0], f32::from(i16::MIN) / f32::from(i16::MAX));
        assert_eq!(planar[1][1], -1.0 / f32::from(i16::MAX));
    }

    #[test]
    fn converts_s24_to_planar_f32() {
        let mut buf = AudioBuffer::<i24>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        buf.resize_uninit(2);
        buf.plane_mut(0)
            .unwrap()
            .copy_from_slice(&[i24::MAX, i24::MIN]);
        let planar = convert_buffer_to_planar_f32(&GenericAudioBufferRef::S24(&buf)).unwrap();
        assert_eq!(planar[0][0], 1.0);
        assert_eq!(planar[0][1], -8_388_608.0 / 8_388_607.0);
    }

    #[test]
    fn converts_s32_to_planar_f32() {
        let mut buf = AudioBuffer::<i32>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        buf.resize_uninit(2);
        buf.plane_mut(0).unwrap().copy_from_slice(&[0, i32::MAX]);
        let planar = convert_buffer_to_planar_f32(&GenericAudioBufferRef::S32(&buf)).unwrap();
        assert_eq!(planar[0][0], 0.0);
        assert_eq!(planar[0][1], 1.0);
    }

    #[test]
    fn converts_floats_to_planar_f32() {
        let mut f32_buf = AudioBuffer::<f32>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        f32_buf.resize_uninit(2);
        f32_buf.plane_mut(0).unwrap().copy_from_slice(&[0.25, -0.5]);
        let planar = convert_buffer_to_planar_f32(&GenericAudioBufferRef::F32(&f32_buf)).unwrap();
        assert_eq!(planar[0], vec![0.25, -0.5]);

        let mut f64_buf = AudioBuffer::<f64>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 2);
        f64_buf.resize_uninit(2);
        f64_buf.plane_mut(0).unwrap().copy_from_slice(&[0.25, -0.5]);
        let planar = convert_buffer_to_planar_f32(&GenericAudioBufferRef::F64(&f64_buf)).unwrap();
        assert_eq!(planar[0], vec![0.25, -0.5]);
    }

    #[test]
    fn rejects_unsupported_buffer_format() {
        let mut buf = AudioBuffer::<u16>::new(AudioSpec::new(44_100, Channels::Discrete(1)), 1);
        buf.resize_uninit(1);
        buf.plane_mut(0).unwrap().copy_from_slice(&[0u16]);
        assert!(convert_buffer_to_planar_f32(&GenericAudioBufferRef::U16(&buf)).is_err());
    }
}
