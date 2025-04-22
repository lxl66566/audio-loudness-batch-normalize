# Audio Loudness Batch Normalize

English | [简体中文](./README-zh_CN.md)

(Previous attempt: [Loudness-Normalization-tauri-app](https://github.com/lxl66566/Loudness-Normalization-tauri-app). As the EBU R128 algorithm proved less effective for galgame voice streaming scenarios, I try to directly normalize the loudness of raw galgame audio instead.)

_Audio Loudness Batch Normalize_ is a command-line tool designed for batch processing audio files to achieve consistent loudness levels across multiple files. It utilizes the EBU R128 algorithm for loudness adjustment and offers configurable target loudness values and true peak limiting.

## Features

- Batch processing with multi-threading
- Configurable true peak limiting
- Automatic target loudness (LUFS) setting

## Installation

Choose one of the following methods:

- Download the latest pre-built binary from [Releases](https://github.com/lxl66566/audio-loudness-batch-normalize/releases).
- Install via [cargo-binstall](https://github.com/cargo-bins/cargo-binstall):
  ```bash
  cargo binstall audio-loudness-batch-normalize
  ```

## Usage

```bash
loudness-normalize <input_directory> [options]
```

Options:

```
<input_directory>                     Input directory for audio files
-o, --output <path>                   Output directory (default: override input audios)
-s, --sample-percentage <float>       Percentage of files to sample (default: 1.00, samples all audio files)
-t, --trim-percentage <float>         Trimming percentage for averaging (default: 0.30, removes the highest 15% and lowest 15%)
    --target-lufs <float>             Target loudness value (LUFS) (default: auto-calculated)
    --true-peak-db <float>            Target true peak (dBTP) (default: -1.5)
-t, --threads <number>                Number of processing threads (default: CPU core count)
```

## Tip

- Only WAV, OGG encoders are supported currently. If the input format is not OGG, the output will be in WAV format and may need to be manually converted.

## Theory

1. Search for all audio files in the input directory.
2. Randomly sample all audio files based on `sample_percentage` to calculate the target loudness (LUFS).
   - If the audio's active duration is too short, the EBU R128 algorithm may return NaN, in which case it will fall back to RMS loudness calculation.
3. Calculate the trimmed mean of audio loudness based on `trim_percentage`. For example, setting `trim_percentage` to 0.3 will remove the loudness values in the top 15% and bottom 15% of the samples.
4. Adjust the audio loudness using the target loudness (LUFS) and the EBU R128 algorithm.

## Special Thanks

Gemini 2.5 Pro Preview 03-25
