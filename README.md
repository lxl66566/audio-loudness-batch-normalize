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

## Options

```
<input_directory>                Input directory containing audio files
-o, --output <path>              Output directory (default: {input_directory}_normalized)
-s, --sample-percentage <float>  Percentage of files to sample (default: 1.00)
-t, --trim-percentage <float>    Trim percentage for loudness calculation (default: 0.30)
    --target-lufs <float>        Target loudness value in LUFS (default: auto-calculated)
    --true-peak-db <float>       Target true peak in dBTP (default: -1.5)
-t, --threads <number>           Number of processing threads (default: CPU core count)
```

## Theory

1. Scans the input directory for all audio files.
2. Samples files based on `sample_percentage` and calculates the trimmed mean loudness using `trim_percentage`. For example, setting `trim_percentage` to 0.3 discards the top 15% and bottom 15% loudness segments in the samples.
3. Adjusts audio loudness using the target LUFS value and the EBU R128 algorithm.

## Special Thanks

Gemini 2.5 Pro Preview 03-25
