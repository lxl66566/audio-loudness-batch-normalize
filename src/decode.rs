use std::{fs, io, path::Path};

use log::warn;
use symphonia::core::{
    audio::{Audio, GenericAudioBufferRef},
    codecs::{
        CodecParameters,
        audio::{AudioDecoder, AudioDecoderOptions},
    },
    errors::Error as SymphoniaError,
    formats::{FormatOptions, FormatReader, TrackType, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

use crate::error::DecodeError;

/// An opened audio file with a decoder prepared for its default audio track.
pub struct DecodedAudio {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    /// Sample rate in Hz of the default audio track.
    pub sample_rate: u32,
    /// Number of channels of the default audio track.
    pub channels: u16,
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
    let decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())?;

    Ok(DecodedAudio {
        format,
        decoder,
        track_id,
        sample_rate,
        channels,
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
const MAX_S24: f32 = 8_388_607.0;

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
