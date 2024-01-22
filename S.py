import numpy as np
import matplotlib.pyplot as plt

from utils import T2F, lpf

import matplotlib
matplotlib.rc("font", family='FangSong')
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_ask_signal(binary_input):
    # 参数定义
    num_bits = len(binary_input)
    sampling_rate = 500
    total_samples = num_bits * sampling_rate
    time_vector = np.linspace(0, num_bits, total_samples)
    carrier_frequency = 8

    # 生成方波信号
    signal_waveform = np.zeros(total_samples)
    for bit_index in range(num_bits):
        if binary_input[bit_index] < 1:
            signal_waveform[bit_index * sampling_rate : (bit_index + 1) * sampling_rate] = 0
        else:
            signal_waveform[bit_index * sampling_rate : (bit_index + 1) * sampling_rate] = 1

    # 绘制原始方波信号
    plt.figure(figsize=(8, 6))
    plt.subplot(5, 1, 1)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, signal_waveform, linewidth=1)
    plt.title('原始方波信号')

    # 生成载波信号
    carrier_wave = np.cos(2 * np.pi * carrier_frequency * time_vector)

    # 绘制载波信号
    plt.subplot(5, 1, 2)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, carrier_wave, linewidth=1)
    plt.title('载波信号')

    # 生成ASK调制信号
    modulated_signal = signal_waveform * carrier_wave

    # 添加高斯白噪声
    noise_amplitude = 0.1  # 调整噪声幅度
    noise = np.random.normal(0, noise_amplitude, total_samples)
    modulated_signal = modulated_signal + noise

    # 绘制ASK调制信号
    plt.subplot(5, 1, 3)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, modulated_signal, linewidth=1)
    plt.title('2ASK 调制信号')

    # 相干解调
    demodulated_signal = modulated_signal * np.cos(2 * np.pi * carrier_frequency * time_vector)
    demodulated_signal = demodulated_signal - np.mean(demodulated_signal)

    # 低通滤波
    freq_samples, freq_spectrum = T2F(time_vector, demodulated_signal)
    time_samples, time_signal = lpf(freq_samples, freq_spectrum, 2 * carrier_frequency)

    # 绘制低通滤波后的信号
    plt.subplot(5, 1, 4)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_samples, time_signal, linewidth=1)
    plt.title('低通滤波后的信号')

    # 抽样判决
    for m in range(num_bits):
        if (time_signal[m * sampling_rate + 250] + 0.5) < 0.5:
            demodulated_signal[m * sampling_rate : (m + 1) * sampling_rate] = 0
        else:
            demodulated_signal[m * sampling_rate : (m + 1) * sampling_rate] = 1

    # 绘制低通滤波后的信号
    plt.subplot(5, 1, 5)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_samples, demodulated_signal, linewidth=1)
    plt.title('抽样判决后的信号')

    plt.tight_layout()
    plt.savefig('temp/ASK.png')
    #plt.show()

def generate_fsk_signal(binary_input):
    # 参数定义
    num_bits = len(binary_input)
    sampling_rate = 500
    total_samples = num_bits * sampling_rate
    time_vector = np.linspace(0, num_bits, total_samples)
    carrier1_frequency = 10     # 载波信号1频率
    carrier2_frequency = 5      # 载波信号2频率
    Baseband_frequency = 2      # 基带信号频率

    # 生成基带信号
    signal_waveform = np.zeros(total_samples)
    for bit_index in range(num_bits):
        if binary_input[bit_index] < 1:
            signal_waveform[bit_index * sampling_rate : (bit_index + 1) * sampling_rate] = 0
        else:
            signal_waveform[bit_index * sampling_rate : (bit_index + 1) * sampling_rate] = 1

    # 绘制原始方波信号
    plt.figure(figsize=(8, 6))
    plt.subplot(6, 1, 1)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, signal_waveform, linewidth=1)
    plt.title('基带信号')

    # 基带信号取反
    st1 = signal_waveform
    st2 = np.logical_not(signal_waveform).astype(int)

    # 载波信号
    s1 = np.cos(2 * np.pi * carrier1_frequency * time_vector)
    s2 = np.cos(2 * np.pi * carrier2_frequency * time_vector)

    # 绘制载波1的波形
    plt.subplot(6, 1, 2)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, s1, linewidth=1)
    plt.title('载波1的波形')

    # 绘制载波2的波形
    plt.subplot(6, 1, 3)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, s2, linewidth=1)
    plt.title('载波2的波形')

    # 调制
    F1 = st1 * s1  # 加入载波1
    F2 = st2 * s2  # 加入载波2
    e_fsk = F1 + F2

    # 添加高斯白噪声到调制信号
    noise_amplitude_modulation = 0.1  # 调整噪声幅度
    noise_modulation = np.random.normal(0, noise_amplitude_modulation, total_samples)
    e_fsk = e_fsk + noise_modulation

    plt.subplot(6, 1, 4)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, e_fsk, linewidth=1)
    plt.title('FSK 调制信号（含噪声）')

    # 相干解调
    st1 = e_fsk * s1  # 与载波1相乘
    f, sf1 = T2F(time_vector, st1)
    _, st1 = lpf(f, sf1, 2 * Baseband_frequency)
    st2 = e_fsk * s2  # 与载波2相乘
    f, sf2 = T2F(time_vector, st2)
    _, st2 = lpf(f, sf2, 2 * Baseband_frequency)

    # 绘制相干解调后的波形
    plt.subplot(6, 1, 5)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, st1 + st2, linewidth=1)
    plt.title('相干解调后的波形')

    # 抽样判决
    at = np.zeros_like(e_fsk)
    for m in range(num_bits):
        if st1[m * sampling_rate + 250] > st2[m * sampling_rate + 250]:
            at[m * sampling_rate : (m + 1) * sampling_rate] = 1
        else:
            at[m * sampling_rate : (m + 1) * sampling_rate] = 0

    plt.subplot(6, 1, 6)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, at, linewidth=1)
    plt.title('抽样判决后的信号')

    plt.tight_layout()
    plt.savefig('temp/FSK.png')
    #plt.show()

def generate_psk_signal(binary_input):
    # 参数定义
    num_bits = len(binary_input)
    sampling_rate = 500
    total_samples = num_bits * sampling_rate
    time_vector = np.linspace(0, num_bits, total_samples)
    carrier_frequency = 5      # 载波信号频率
    Baseband_frequency = 2     # 基带信号频率
    Bandwidth = 2 * Baseband_frequency

    # 生成基带信号
    signal_waveform1 = np.zeros(total_samples)
    for bit_index in range(num_bits):
        if binary_input[bit_index] < 1:
            signal_waveform1[bit_index * sampling_rate : (bit_index + 1) * sampling_rate] = 0
        else:
            signal_waveform1[bit_index * sampling_rate : (bit_index + 1) * sampling_rate] = 1

    # 绘制原始基带信号
    plt.figure(figsize=(8, 6))
    plt.subplot(6, 1, 1)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, signal_waveform1, linewidth=1)
    plt.title('原始基带信号')

    # 基带信号求反
    signal_waveform2 = np.logical_not(signal_waveform1).astype(int)

    # 基带信号变成双极性
    signal_waveform3 = signal_waveform1 - signal_waveform2

    # 绘制双极性基带信号
    plt.subplot(6, 1, 2)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, signal_waveform3, linewidth=1)
    plt.title('基带信号变成双极性')

    # 载波信号
    carrier_wave = np.sin(2 * np.pi * carrier_frequency * time_vector)

    # 绘制载波信号
    plt.subplot(6, 1, 3)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, carrier_wave, linewidth=1)
    plt.title('载波信号')

    # 调制
    modulated_signal = signal_waveform3 * carrier_wave

    # 添加高斯白噪声到调制信号
    noise_amplitude_modulation = 0.1  # 调整噪声幅度
    noise_modulation = np.random.normal(0, noise_amplitude_modulation, total_samples)
    modulated_signal = modulated_signal + noise_modulation

    # 绘制PSK调制信号（含噪声）
    plt.subplot(6, 1, 4)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, modulated_signal, linewidth=1)
    plt.title('PSK 调制信号（含噪声）')

    # 相干解调
    modulated_signal = modulated_signal * carrier_wave
    f, af = T2F(time_vector, modulated_signal)
    _, modulated_signal = lpf(f, af, Bandwidth)

    # 添加子图，绘制相干解调后的波形
    plt.subplot(6, 1, 5)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, modulated_signal, linewidth=1)
    plt.title('相干解调后的波形')

    # 抽样判决
    sampled_decisions = np.zeros_like(modulated_signal)
    for m in range(num_bits):
        if modulated_signal[m * sampling_rate + 250] < 0:
            sampled_decisions[m * sampling_rate : (m + 1) * sampling_rate] = 0
        else:
            sampled_decisions[m * sampling_rate : (m + 1) * sampling_rate] = 1

    # 绘制抽样判决后的信号
    plt.subplot(6, 1, 6)
    plt.xticks(np.arange(0, num_bits+1, 1))
    plt.plot(time_vector, sampled_decisions, linewidth=1)
    plt.title('抽样判决后的信号')

    plt.tight_layout()
    plt.savefig('temp/PSK.png')
    #plt.show()

