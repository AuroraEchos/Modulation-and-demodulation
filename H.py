import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from commpy.modulation import QAMModem
matplotlib.rc("font", family='FangSong')
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_qpsk_signal():
    # 参数定义
    num_bits = 20               # 比特数
    bit_period = 1              # 比特周期
    carrier_frequency = 2       # 载波频率
    sampling_frequency = 100    # 抽样频率
    bitstream = np.random.choice([-1, 1], size=num_bits)  # 随机产生的比特数-1或1
    i_channel = bitstream[::2]      # 奇数位进入 I 路
    q_channel = bitstream[1::2]     # 偶数位进入 Q 路

    # 生成比特数据
    bit_data = np.repeat(bitstream, bit_period * sampling_frequency)

    # 生成 I 路和 Q 路的数据
    i_data = np.repeat(i_channel, 2 * bit_period * sampling_frequency)
    q_data = np.repeat(q_channel, 2 * bit_period * sampling_frequency)

    t = np.arange(0, num_bits * bit_period, 1 / sampling_frequency)

    qpsk_insignal(t, bit_data, i_data, q_data)


    # 载波信号
    bit_t = np.arange(0, 2 * bit_period, 1 / sampling_frequency)
    I_carrier = np.concatenate([i * np.cos(2 * np.pi * carrier_frequency * bit_t) for i in i_channel])
    Q_carrier = np.concatenate([q * np.cos(2 * np.pi * carrier_frequency * bit_t + np.pi / 2) for q in q_channel])

    # 传输信号
    QPSK_signal = I_carrier + Q_carrier

    qpsk_resignal(t, QPSK_signal, I_carrier, Q_carrier)


    # 接收信号
    snr = 1                     
    QPSK_receive = QPSK_signal + np.random.normal(0, np.sqrt(snr), len(QPSK_signal))

    qpsk_nosignal(t, QPSK_receive)

    # 解调
    I_recover = np.zeros(num_bits // 2)
    Q_recover = np.zeros(num_bits // 2)

    # 初始化前一比特的值
    prev_I_decision = 0
    prev_Q_decision = 0

    for i in range(num_bits // 2):
        I_output = QPSK_receive[(i * len(bit_t)) : ((i + 1) * len(bit_t))] * np.cos(2 * np.pi * carrier_frequency * bit_t)
        Q_output = QPSK_receive[(i * len(bit_t)) : ((i + 1) * len(bit_t))] * np.cos(2 * np.pi * carrier_frequency * bit_t + np.pi / 2)

        # 判决反馈：使用前一比特的信息来辅助当前比特的判决
        I_sum = np.sum(I_output) + prev_I_decision
        Q_sum = np.sum(Q_output) + prev_Q_decision

        # 更新前一比特的信息
        prev_I_decision = np.sign(I_sum)
        prev_Q_decision = np.sign(Q_sum)

        # 判决当前比特
        I_recover[i] = prev_I_decision
        Q_recover[i] = prev_Q_decision

    # 并/串变换
    bit_recover = np.zeros(num_bits)
    for i in range(num_bits):
        if i % 2 !=0:
            bit_recover[i] = I_recover[(i - 1) // 2]
        else:
            bit_recover[i] = Q_recover[i // 2]
    
    # 适用绘图比较 I、Q 比特流
    recover_data = np.repeat(bit_recover, bit_period * sampling_frequency)
    I_recover_data = np.repeat(I_recover, 2 * bit_period * sampling_frequency)
    Q_recover_data = np.repeat(Q_recover, 2 * bit_period * sampling_frequency)

    qpsk_designal(t, recover_data, I_recover_data, Q_recover_data)

    qpsk_cosignal(t, bit_data, recover_data, i_data, q_data, I_recover_data, Q_recover_data)

def generate_qam_signal(M):
    # 参数定义
    M = M
    mqam_count = 400
    snr = 13

    # 生成随机输入信号
    bit_per_symbol = int(np.log2(M))
    bit_length = mqam_count * bit_per_symbol
    bit_signal = np.round(np.random.rand(bit_length)).astype(int)
    

    # QAM调制
    qam_modem = QAMModem(M)
    qam_symbols = qam_modem.modulate(bit_signal)

    # 添加高斯白噪声
    noise_real = np.random.randn(len(qam_symbols))
    noise_imag = np.random.randn(len(qam_symbols))
    noise = np.vectorize(complex)(noise_real, noise_imag)
    snr_linear = 10**(snr / 10.0)
    signal_with_noise = qam_symbols + np.sqrt(1 / snr_linear) * noise

    # 模拟误码率曲线
    snr_qq = 0
    qq = np.zeros(150)
    for i in range(150):
        bit_moded_qq = qam_modem.modulate(bit_signal)
        snr_qq += 0.1
        signal_time_C_W_R = np.random.normal(np.real(bit_moded_qq), scale=np.sqrt(1 / 10**(snr_qq / 10.0)))
        signal_time_C_W_i = np.random.normal(np.imag(bit_moded_qq), scale=np.sqrt(1 / 10**(snr_qq / 10.0)))
        signal_time_C_W = signal_time_C_W_R + 1j * signal_time_C_W_i
        bit_demod_sig = qam_modem.demodulate(signal_time_C_W, 'hard')
        error_bit = np.sum(bit_demod_sig != bit_signal)
        error_rate = error_bit / len(bit_signal)
        qq[i] = error_rate


    Inputsignal_image(bit_signal, M)
    Outputsignal_image(bit_demod_sig, M)
    Constellation_image(qam_symbols, signal_with_noise, M)
    Ber_image(qq, M)
    Powerspectrum_image(qam_symbols, signal_with_noise, M)
    plot_all_images(bit_signal, qam_symbols, signal_with_noise, bit_demod_sig, qq, M)

def Inputsignal_image(bit_signal, M):
    # 绘制输入信号图
    plt.figure(figsize=(8, 6))
    plt.plot(bit_signal,linewidth = 1)
    plt.axis([0, 100, -1, 2])
    plt.title('发射码元')
    plt.xlabel('t')
    plt.ylabel('二进制值')
    plt.savefig(f'temp/QAM_{M}_input.png')

def Constellation_image(qam_symbols, signal_with_noise, M):
    # 绘制星座图
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), c='red', marker='.', s=5)
    plt.title('调制后的星座图')

    plt.subplot(2, 1, 2)
    plt.scatter(1.5 * np.real(signal_with_noise), 1.5 * np.imag(signal_with_noise), c='red', marker='.', s=5)
    plt.title('接收的星座图')

    plt.savefig(f'temp/QAM_{M}_constellation.png')

def Outputsignal_image(bit_demod_sig, M):
    # 绘制解调信号
    plt.figure(figsize=(8, 6))
    plt.plot(bit_demod_sig,linewidth = 1)
    plt.axis([0, 100, -1, 2])
    plt.title('接收码元')
    plt.xlabel('t')
    plt.ylabel('二进制值')
    plt.savefig(f'temp/QAM_{M}_output.png')

def Ber_image(qq, M):
    # 绘制误码率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, 15, 0.1), qq, linewidth=1)
    plt.title('误码率曲线')
    plt.xlabel('信噪比（dB）')
    plt.ylabel('误码率')
    plt.savefig(f'temp/QAM_{M}_ber.png')
    #plt.gca().invert_yaxis()

def Powerspectrum_image(qam_symbols, signal_with_noise, M):
    # 绘制功率谱密度
    plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 1)
    f_tx, power_spectrum_tx = signal.welch(qam_symbols, fs=1, nperseg=256, return_onesided=False)
    plt.plot(f_tx, 10 * np.log10(power_spectrum_tx), linewidth=1)
    plt.title('发射时功率谱')
    #plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')

    plt.subplot(2, 1, 2)
    f_rx, power_spectrum_rx = signal.welch(signal_with_noise, fs=1, nperseg=256, return_onesided=False)
    plt.plot(f_rx, 10 * np.log10(power_spectrum_rx), linewidth=1, color='green')
    plt.title('接收时功率谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')

    plt.tight_layout()
    plt.savefig(f'temp/QAM_{M}_powerspectrum.png')

def plot_all_images(bit_signal, qam_symbols, signal_with_noise, bit_demod_sig, qq, M):
    plt.figure(figsize=(8, 6))

    # 发射码元图
    plt.subplot(3, 2, 1)
    plt.plot(bit_signal, linewidth=1)
    plt.axis([0, 100, -1, 2])
    plt.title('发射码元')
    plt.xlabel('t')
    plt.ylabel('二进制值')

    # 调制后星座图和接收后星座图
    plt.subplot(3, 2, 3)
    plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), c='red', marker='.', s=5)
    plt.title('调制后的星座图')

    plt.subplot(3, 2, 4)
    plt.scatter(1.5 * np.real(signal_with_noise), 1.5 * np.imag(signal_with_noise), c='red', marker='.', s=5)
    plt.title('接收的星座图')

    # 接收码元图
    plt.subplot(3, 2, 2)
    plt.plot(bit_demod_sig, linewidth=1)
    plt.axis([0, 100, -1, 2])
    plt.title('接收码元')
    plt.xlabel('t')
    plt.ylabel('二进制值')

    # 误码率曲线
    plt.subplot(3, 2, 6)
    plt.plot(np.arange(0, 15, 0.1), qq, linewidth=1)
    plt.title('误码率曲线')
    plt.xlabel('信噪比（dB）')
    plt.ylabel('误码率')

    # 发射和接收时功率谱密度
    plt.subplot(3, 2, 5)
    f_tx, power_spectrum_tx = signal.welch(qam_symbols, fs=1, nperseg=256, return_onesided=False)
    plt.plot(f_tx, 10 * np.log10(power_spectrum_tx), linewidth=1, label='发射时')
    
    f_rx, power_spectrum_rx = signal.welch(signal_with_noise, fs=1, nperseg=256, return_onesided=False)
    plt.plot(f_rx, 10 * np.log10(power_spectrum_rx), linewidth=1, label='接收时')
    
    plt.title('发射和接收时功率谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'temp/QAM_{M}_all.png')
    #plt.show()

def qpsk_insignal(t, bit_data, i_data, q_data):
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, bit_data,linewidth = 1)
    plt.title('输入信号')

    plt.subplot(3, 1, 2)
    plt.plot(t, i_data, linewidth = 1)
    plt.title('I 路信号')

    plt.subplot(3, 1, 3)
    plt.plot(t, q_data, linewidth = 1)
    plt.title('Q 路信号')

    plt.tight_layout()
    plt.savefig(f'temp/QPSK_in.png')

def qpsk_resignal(t, QPSK_signal, I_carrier, Q_carrier):
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, QPSK_signal, linewidth=1)
    plt.title('QPSK 已调信号')

    plt.subplot(3, 1, 2)
    plt.plot(t, I_carrier, linewidth=1)
    plt.title('I 路已调信号')

    plt.subplot(3, 1, 3)
    plt.plot(t, Q_carrier, linewidth=1)
    plt.title('Q 路已调信号')

    plt.tight_layout()
    plt.savefig(f'temp/QPSK_re.png')

def qpsk_designal(t, recover_data, I_recover_data, Q_recover_data):
    plt.figure(figsize=(8, 6))

    # 绘制并/串变换后的波形
    plt.subplot(3, 1, 1)
    plt.plot(t, recover_data, linewidth=1)
    plt.title('并/串变换后的波形')

    # 绘制并/串变换后的 I 路波形
    plt.subplot(3, 1, 2)
    plt.plot(t, I_recover_data, linewidth=1)
    plt.title('并/串变换后的 I 路波形')

    # 绘制并/串变换后的 Q 路波形
    plt.subplot(3, 1, 3)
    plt.plot(t, Q_recover_data, linewidth=1)
    plt.title('并/串变换后的 Q 路波形')

    plt.tight_layout()
    plt.savefig('temp/QPSK_de.png')

def qpsk_nosignal(t, QPSK_receive):
    plt.figure(figsize=(8, 6))
    
    # 绘制接收信号波形
    plt.plot(t, QPSK_receive, linewidth=1)
    plt.title('接收信号波形')
    plt.xlabel('时间')
    plt.ylabel('幅度')
    
    plt.tight_layout()
    plt.savefig('temp/QPSK_no.png')

def qpsk_cosignal(t, bit_data, recover_data, i_data, q_data, I_recover_data, Q_recover_data):
    plt.figure(figsize=(8, 6))

    plt.subplot(3, 4, 5)
    plt.plot(t, bit_data, linewidth=1)
    plt.title('输入')

    plt.subplot(3, 4, 8)
    plt.plot(t, recover_data, linewidth=1)
    plt.title('输出')

    plt.subplot(3, 4, 2)
    plt.plot(t, i_data, linewidth=1)
    plt.title('I 路输入')

    plt.subplot(3, 4, 3)
    plt.plot(t, I_recover_data, linewidth=1)
    plt.title('I 路输出')

    plt.subplot(3, 4, 10)
    plt.plot(t, q_data, linewidth=1)
    plt.title('Q 路输入')

    plt.subplot(3, 4, 11)
    plt.plot(t, Q_recover_data, linewidth=1)
    plt.title('Q 路输出')

    plt.yticks([-1, 1])
    plt.tight_layout()
    plt.savefig('temp/QPSK_co.png')