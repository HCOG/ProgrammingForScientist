"""
Lesson 01: Fourier Transform — 从频率视角看世界
================================================
目标：理解如何把时域信号拆解成频率成分
知识点：傅里叶变换、频谱分析、信号处理

学习目标（反向阅读提示）：
- 先看最后的主程序，观察输入输出
- 倒推：为什么采样率影响最高可分析频率？
- 倒推：FFT输出数组的索引对应什么频率？
"""

import numpy as np


def generate_composite_signal(t, frequencies, amplitudes, phases):
    """
    生成复合正弦信号
    
    参数:
        t: 时间数组
        frequencies: 频率列表 [Hz]
        amplitudes: 振幅列表
        phases: 初相位列表 [弧度]
    
    返回:
        composite: 复合信号
    """
    composite = np.zeros_like(t)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        composite += amp * np.sin(2 * np.pi * freq * t + phase)
    return composite


def compute_fft(signal, sample_rate):
    """
    计算快速傅里叶变换并返回正频率部分
    
    关键洞察：FFT输出是复数数组
    - 第一个元素是直流分量（频率为0）
    - 第i个元素对应频率 i * sample_rate / N Hz
    """
    N = len(signal)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, 1 / sample_rate)
    
    # 取正频率部分（对称性：负频率部分与正频率共轭）
    positive_mask = frequencies >= 0
    return frequencies[positive_mask], np.abs(fft_result[positive_mask]) * 2 / N


def find_dominant_frequencies(freqs, magnitudes, top_n=3):
    """找出幅值最大的N个频率成分"""
    # 排除直流分量（index 0）
    magnitudes_no_dc = magnitudes.copy()
    magnitudes_no_dc[0] = 0
    top_indices = np.argsort(magnitudes_no_dc)[-top_n:][::-1]
    return [(freqs[i], magnitudes[i]) for i in top_indices]


if __name__ == "__main__":
    # ===== 参数设置 =====
    DURATION = 1.0          # 信号持续时间 [秒]
    SAMPLE_RATE = 1000      # 采样率 [Hz] — 决定最高可分析频率
    TRUE_FREQUENCIES = [5, 15, 50]  # 隐藏的真实频率
    
    # ===== 生成时间轴 =====
    t = np.arange(0, DURATION, 1 / SAMPLE_RATE)
    
    # ===== 构造复合信号 =====
    amplitudes = [1.0, 0.5, 0.3]
    phases = [0, np.pi / 4, np.pi / 2]
    signal = generate_composite_signal(t, TRUE_FREQUENCIES, amplitudes, phases)
    
    # ===== 加入噪声（模拟真实测量） =====
    np.random.seed(42)
    noisy_signal = signal + 0.1 * np.random.randn(len(signal))
    
    # ===== 频谱分析 =====
    freqs, spectrum = compute_fft(noisy_signal, SAMPLE_RATE)
    
    # ===== 找出主频率 =====
    dominant = find_dominant_frequencies(freqs, spectrum, top_n=3)
    
    # ===== 输出结果 =====
    print("=" * 50)
    print("傅里叶变换分析结果")
    print("=" * 50)
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"信号时长: {DURATION} 秒")
    print(f"采样点数: {len(t)}")
    print(f"最高可分析频率（Nyquist）: {SAMPLE_RATE / 2} Hz")
    print()
    print("检测到的主频率成分:")
    for i, (f, mag) in enumerate(dominant, 1):
        print(f"  #{i}: 频率 = {f:.1f} Hz, 幅值 = {mag:.3f}")
    print()
    print("对比真实频率:", TRUE_FREQUENCIES)
