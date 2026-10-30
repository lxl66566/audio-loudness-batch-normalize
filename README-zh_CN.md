# Audio Loudness Batch Normalize

[English](./README.md) | 简体中文

（我的早期尝试：[Loudness-Normalization-tauri-app](https://github.com/lxl66566/Loudness-Normalization-tauri-app)。由于 EBU R128 算法在 galgame 的流式人声场景中效果低于预期，因此我尝试直接对 galgame 原音频进行响度归一化调整。）

_Audio Loudness Batch Normalize_ 是一个命令行工具，用于批量处理音频文件以实现多个文件之间的一致响度水平。它使用 EBU R128 算法进行响度调整，并提供了可配置的目标响度值和真峰值限制。

## Features

- 批量多线程处理
- 可配置的真峰值限制
- 自动目标响度（LUFS）设置

## Installation

任选其一：

- 在 [Releases](https://github.com/lxl66566/audio-loudness-batch-normalize/releases) 中下载最新版本的二进制文件。
- 使用 [cargo-binstall](https://github.com/cargo-bins/cargo-binstall) 安装：
  ```bash
  cargo binstall audio-loudness-batch-Normalize
  ```

## Usage

```bash
loudness-normalize <输入目录> [选项]
```

选项：

```
<输入目录>                       输入音频文件目录
-o, --output <路径>              输出目录（默认：直接修改输入音频文件）
-s, --sample-percentage <浮点数> 采样文件的百分比（默认：1.00，将采样所有音频文件）
    --trim-percentage <浮点数>   计算平均时的截尾百分比（默认：0.30，会截掉最高 15% 和最低 15% 的部分）
    --target-lufs <浮点数>       目标响度值（LUFS）（默认：自动计算）
    --true-peak-db <浮点数>      目标真峰值（dBTP）（默认：-1.5）
    --wav-bit-depth <格式>       WAV 输出采样格式：auto、s16、s24、s32、f32（默认：auto）
    --ogg-quality <浮点数>       Ogg Vorbis 感知质量（默认：5.0，范围 0.0 到 10.0）
-t, --threads <数字>             处理线程数（默认：CPU核心数）
-n, --measure-only               仅测量并计算目标响度（LUFS），不执行响度变更
```

## Tip

- 当前仅支持 WAV、OGG 的编码器，如果输入格式不是 OGG 格式，输出将会是 WAV 格式，原文件将被原样保留。如果你需要让均衡后的音频保持原先的格式，需要手动转换。
- WAV 输出采样格式默认跟随无损整型源（8/16-bit 输出 16-bit，24-bit 保持 24-bit，32-bit 保持 32-bit）；浮点源和有损源输出 32-bit float。可用 `--wav-bit-depth` 覆盖。
- Ogg Vorbis 输出默认使用感知质量 5 编码，可用 `--ogg-quality` 调整。
- 格式转换时音频元数据（tags）不会被保留。

## Theory

1. 在输入目录中查找所有音频文件。
2. 根据 sample_percentage 进行所有音频的随机采样，以计算目标响度（LUFS）。
   - 如果音频有声时间过短，使用 EBU R128 算法可能会算出 NaN 值，此时将 fallback 到 RMS 响度计算。
3. 依据 trim_percentage 进行音频响度截尾均值计算。例如，将 trim_percentage 设为 0.3，则会在样本中去除响度在最高 15% 和最低 15% 的部分。
4. 使用目标响度（LUFS）和 EBU R128 算法进行音频响度调整。
