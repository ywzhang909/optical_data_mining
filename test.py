import numpy as np
import matplotlib.pyplot as plt

# 模拟数据生成：来回震荡的坐标值（例如弹簧振子）
def generate_oscillation_data(freq=5, duration=10, sampling_rate=100):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    x = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.normal(size=len(t))  # 添加噪声
    return t, x

# 傅里叶变换分析函数
def analyze_fft(t, x):
    n = len(x)
    y_fft = np.fft.fft(x)
    fft_magnitude = np.abs(y_fft)[:n//2]  # 只取前半部分（对称）
    
    # 频率轴
    fs = 1 / (t[1] - t[0])
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]

    # 找出最大能量对应的频率
    dominant_freq = freqs[np.argmax(fft_magnitude)]
    
    return freqs, fft_magnitude, dominant_freq

# 绘图展示
def plot_fft_results(t, x, freqs, magnitudes, dominant_freq):
    plt.figure(figsize=(12,6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title("Time Domain Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitudes)
    plt.title("Frequency Domain (FFT Magnitude)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.axvline(dominant_freq, color='r', linestyle='--', label=f'Dominant Frequency: {dominant_freq:.2f} Hz')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 主程序流程
if __name__ == "__main__":
    # Step 1: 生成模拟数据
    t, x = generate_oscillation_data(freq=5, duration=10, sampling_rate=100)

    # Step 2: 进行 FFT 分析
    freqs, magnitudes, dominant_freq = analyze_fft(t, x)

    # Step 3: 打印结果
    print(f"检测到的主要频率: {dominant_freq:.2f} Hz")

    # Step 4: 绘图展示
    plot_fft_results(t, x, freqs, magnitudes, dominant_freq)