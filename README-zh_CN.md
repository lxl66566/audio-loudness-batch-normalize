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
-o, --output <路径>              输出目录（默认：{输入目录}_normalized）
-s, --sample-percentage <浮点数> 采样文件的百分比（默认：1.00，将采样所有音频文件）
-t, --trim-percentage <浮点数>   计算平均时的截尾百分比（默认：0.30，会截掉最高 15% 和最低 15% 的部分）
    --target-lufs <浮点数>       目标响度值（LUFS）（默认：自动计算）
    --true-peak-db <浮点数>      目标真峰值（dBTP）（默认：-1.5）
-t, --threads <数字>             处理线程数（默认：CPU核心数）
```

## Theory

1. 在输入目录中查找所有音频文件。
2. 根据 sample_percentage 进行所有音频的随机采样，以计算目标响度（LUFS）。
   - 如果音频有声时间过短，使用 EBU R128 算法可能会算出 NaN 值，此时将 fallback 到 RMS 响度计算。
3. 依据 trim_percentage 进行音频响度截尾均值计算。例如，将 trim_percentage 设为 0.3，则会在样本中去除响度在最高 15% 和最低 15% 的部分。
4. 使用目标响度（LUFS）和 EBU R128 算法进行音频响度调整。

## Special Thanks

Gemini 2.5 Pro Preview 03-25
