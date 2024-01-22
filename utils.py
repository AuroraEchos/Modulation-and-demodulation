import numpy as np

def F2T(f, sf):
    df = f[1] - f[0]
    Fmx = (f[-1] - f[0] + df)
    dt = 1 / Fmx
    N = len(sf)
    T = dt * N
    t = np.arange(0, T, dt)
    sff = np.fft.fftshift(sf)
    st = Fmx * np.fft.ifft(sff)
    return t, st

def T2F(t, st):
    dt = t[1] - t[0]
    T = t[-1]
    df = 1 / T
    N = len(st)
    f = np.arange(-N/2, N/2) * df
    sf = np.fft.fft(st)
    sf = T / N * np.fft.fftshift(sf)
    return f, sf

def lpf(f, sf, B):
    # 该函数使用低通滤波器对输入数据进行滤波
    # 输入: f: 频率样本
    #      sf: 输入数据频谱样本
    #      B: 矩形低通滤波器的带宽
    # 输出: t: 时间样本
    #      st: 输出数据时间样本
    
    df = f[1] - f[0]
    T = 1 / df
    hf = np.zeros(len(f))  # 创建全零矩阵
    bf = np.arange(-np.floor(B/df), np.floor(B/df) + 1) + np.floor(len(f)/2)
    hf[bf.astype(int)] = 1
    yf = hf * sf
    t, st = F2T(f, yf)
    st = np.real(st)
    return t, st

# 计算功率谱
def calculate_power_spectrum(signal, sampling_rate):

    n = len(signal)
    f = np.fft.fftfreq(n, d=1/sampling_rate)
    spectrum = np.fft.fft(signal)
    power_spectrum = np.abs(spectrum)**2 / n

    # 处理 Pxx 为零的情况
    power_spectrum[power_spectrum == 0] = np.finfo(float).eps

    return f, 10 * np.log10(power_spectrum)


def input_binary_sequence():
    binary_sequence = input("请输入二进制序列（例如: 0101101101）: ")
    return np.array([int(bit) for bit in binary_sequence])