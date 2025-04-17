from pathlib import Path

import librosa
import matplotlib.pyplot as plt


def plot_loudness_comparison(audio_path1, audio_path2):
    # 加载音频文件
    y1, sr1 = librosa.load(audio_path1)
    y2, sr2 = librosa.load(audio_path2)

    # 计算响度 (RMS能量)
    S1 = librosa.feature.rms(y=y1) ** 2  # 转换为功率
    S2 = librosa.feature.rms(y=y2) ** 2

    # 转换为绝对分贝值 (使用ref_power=1.0获取绝对dB SPL)
    S1_db = librosa.power_to_db(S1, ref=1.0)
    S2_db = librosa.power_to_db(S2, ref=1.0)

    # 创建时间轴
    times1 = librosa.times_like(S1)
    times2 = librosa.times_like(S2)

    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.plot(times1, S1_db[0], label=f"音频1 ({Path(audio_path1).stem})", alpha=0.8)
    plt.plot(times2, S2_db[0], label=f"音频2 ({Path(audio_path2).stem})", alpha=0.8)

    plt.title("音频绝对响度对比")
    plt.xlabel("时间 (秒)")
    plt.ylabel("响度 (dB SPL)")
    plt.legend()
    plt.grid(True)

    # 保存图片
    plt.savefig(
        f"loudness_comparison_{Path(audio_path1).stem}_{Path(audio_path2).stem}.png"
    )
    plt.close()


if __name__ == "__main__":
    audio_file_dir_1 = r"Z:\voice"
    audio_file_dir_2 = r"Z:\voice_normalized"

    files_1 = list(sorted(Path(audio_file_dir_1).glob("*")))
    files_2 = list(sorted(Path(audio_file_dir_2).glob("*")))

    print(len(files_1), len(files_2))

    for file_1, file_2 in list(zip(files_1, files_2))[:20]:
        plot_loudness_comparison(file_1, file_2)
