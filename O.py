import numpy as np
import random
import scipy.signal
import cv2

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

import matplotlib
matplotlib.rc("font", family='FangSong')
matplotlib.rcParams['axes.unicode_minus'] = False


class OFDM:
    def __init__(self, nFreqSamples=2048, pilotDistanceInSamples=16, pilotAmplitude=2, nData=256, nCyclic=None):
        self.nIFFT = nFreqSamples
        self.nData = nData
        if nCyclic:
            self.nCyclic = nCyclic
        else:
            self.nCyclic = int(self.nIFFT * 2 / 4)

        self.pilot_distance = pilotDistanceInSamples

        self.pilot_amplitude = pilotAmplitude

        self.k_start = int(self.nIFFT - self.nIFFT / self.pilot_distance / 2 - nData * 4 / 2)

    def encode(self, signal, data, randomSeed=1):

        self.spectrum = np.zeros(self.nIFFT, dtype=complex)

        k = self.k_start

        random.seed(randomSeed)

        pilot_counter = self.pilot_distance / 2

        for x in range(self.nData):

            databyte = int(data[x])
            r = int(random.randint(0, 255))
            databyte = int(databyte ^ r)
            bitstream = np.zeros(8)
            for bit in range(8):
                m = 1 << bit
                testbit = m & databyte
                if testbit > 0:
                    bitstream[bit] = 1
                else:
                    bitstream[bit] = -1

            for cnum in range(4):
                pilot_counter = pilot_counter - 1
                if pilot_counter == 0:
                    pilot_counter = self.pilot_distance
                    self.spectrum[k] = self.pilot_amplitude
                    k = k + 1
                    if not (k < self.nIFFT):
                        k = 0
                self.spectrum[k] = complex(bitstream[int(cnum * 2)],
                                           bitstream[int(cnum * 2 + 1)])
                k = k + 1
                if not (k < self.nIFFT):
                    k = 0

        complex_symbol = np.fft.ifft(self.spectrum)
        tx_symbol = np.zeros(len(complex_symbol) * 2)

        s = 1
        txindex = 0
        for smpl in complex_symbol:
            tx_symbol[txindex] = s * np.real(smpl)
            txindex = txindex + 1
            tx_symbol[txindex] = s * np.imag(smpl)
            txindex = txindex + 1
            s = s * -1

        cyclicPrefix = tx_symbol[-self.nCyclic:]

        signal = np.concatenate((signal, cyclicPrefix))
        signal = np.concatenate((signal, tx_symbol))

        return signal
    
    def initDecode(self, signal, offset):
        self.s = 1
        self.rxindex = offset
        self.signal = signal

    def decode(self, randomSeed=1):
        self.rxindex = self.rxindex + self.nCyclic

        rx_symbol = np.zeros(self.nIFFT, dtype=complex)

        for a in range(self.nIFFT):
            realpart = self.s * self.signal[self.rxindex]
            self.rxindex = self.rxindex + 1
            imagpart = self.s * self.signal[self.rxindex]
            self.rxindex = self.rxindex + 1
            rx_symbol[a] = complex(realpart, imagpart)
            self.s = self.s * -1

        isymbol = np.fft.fft(rx_symbol)
        random.seed(randomSeed)

        k = self.k_start

        pilot_counter = self.pilot_distance / 2

        data = np.zeros(self.nData)

        imPilots = 0
        for x in range(self.nData):
            bitstream = np.zeros(8)
            for cnum in range(4):
                pilot_counter = pilot_counter - 1
                if pilot_counter == 0:
                    pilot_counter = self.pilot_distance
                    imPilots = imPilots + np.abs(np.imag(isymbol[k]))
                    k = k + 1
                    if not (k < self.nIFFT):
                        k = 0
                bitstream[int(cnum * 2)] = np.heaviside(np.real(isymbol[k]), 0)
                bitstream[int(cnum * 2 + 1)] = np.heaviside(np.imag(isymbol[k]), 0)
                k = k + 1
                if not (k < self.nIFFT):
                    k = 0

            databyte = 0

            for bit in range(8):
                mask = 1 << bit
                if (bitstream[bit] > 0):
                    databyte = int(mask | int(databyte))

            r = int(random.randint(0, 255))
            databyte = databyte ^ r

            data[x] = databyte

        return data, imPilots
    
    def findSymbolStartIndex(self, signal, searchrangecoarse=None, searchrangefine=25):

        if not searchrangecoarse:
            searchrangecoarse = self.nIFFT * 10
            
        crosscorr = np.array([])
        for i in range(searchrangecoarse):
            s1 = signal[i:i+self.nCyclic]
            s2 = signal[i+self.nIFFT*2:i+self.nIFFT*2+self.nCyclic]
            cc = np.correlate(s1, s2)
            crosscorr = np.append(crosscorr, cc)

        pks, _ = scipy.signal.find_peaks(crosscorr, distance=self.nIFFT*2)
        o1 = pks[0]

        imagpilots = np.array([])
        for i in range(o1 - searchrangefine, o1 + searchrangefine):
            self.initDecode(signal, i)
            _, im = self.decode()
            imagpilots = np.append(imagpilots, im)

        o2 = o1 + np.argmin(imagpilots) - searchrangefine
        return crosscorr, imagpilots, o2
    
def test():
    # 选择测试图像
    image_path = 'temp/grey.png'
    a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ymax, xmax = np.shape(a)

    # 绘制原始图像
    show_orimage(image_path)

    # 初始化
    ofdm = OFDM()
    delay_factor=3

    offset = ofdm.nIFFT * delay_factor
    signal = np.zeros(offset)

    # 对每一行进行编码
    for y in range(ymax):
        row = a[y, :]
        signal = ofdm.encode(signal, row)

    # 将编码后的数据以wav格式保存，模拟含噪信道
    wavfile.write('temp/ofdm.wav', 8000, signal)

    show_spectrum(ofdm, signal)

    # 读取wav作为接收信号
    fs, si = wavfile.read('temp/ofdm.wav')

    # 寻找符号起始索引
    si = np.append(signal, np.zeros(ofdm.nIFFT * 2))
    searchRangeForPilotPeak = 25
    cc, sumofimag, offset = ofdm.findSymbolStartIndex(si, searchrangefine=searchRangeForPilotPeak)

    show_search(cc, offset, searchRangeForPilotPeak, sumofimag)

    # 解码
    ofdm.initDecode(si, offset)
    img = np.empty((ymax, xmax))
    for y in range(ymax):
        row, i = ofdm.decode()
        img[y, :] = row

    show_deimage(img)
    show_coimage(image_path, img)

def show_orimage(image_path):
    plt.figure(figsize=(8, 6))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.axis('off')
    plt.savefig('temp/OFDM_or.png')
    #plt.show()

def show_spectrum(ofdm, signal):
    plt.figure(figsize=(8, 6))

    plt.subplot(3, 1, 1)
    time_domain_signal = np.real(np.fft.ifft(ofdm.spectrum))
    plt.plot(time_domain_signal, linewidth=1)
    plt.title('OFDM符号的时域信号')
    plt.xlabel('时域样本索引')
    plt.ylabel('幅度值')

    plt.subplot(3, 1, 2)
    plt.plot(np.abs(ofdm.spectrum), linewidth=1)
    plt.title('经过OFDM编码后的复数频谱的幅度')

    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0, 1, len(signal)), np.abs(np.fft.fft(signal)) / len(signal), linewidth=1)
    plt.title('发送OFDM的频谱')
    plt.xlabel("归一化频率")
    plt.ylabel("每个频率的信号幅度值")
    plt.tight_layout()

    plt.savefig('temp/OFDM_sp.png')
    #plt.show()

def show_search(cc, offset, searchRangeForPilotPeak, sumofimag):
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(cc)
    plt.axvline(x=offset, color=mcolors.BASE_COLORS['g'])
    plt.title("交叉相关找循环前缀")
    plt.xlabel("信号样本的索引")
    plt.ylabel("信号与移动后的信号进行交叉相关的结果")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(-searchRangeForPilotPeak, searchRangeForPilotPeak), sumofimag)
    plt.title("导频虚部的绝对值之和")
    plt.xlabel("相对于OFDM符号起始位置的样本索引")
    plt.ylabel("导频虚部的绝对值之和")

    plt.tight_layout()
    plt.savefig('temp/OFDM_se.png')
    #plt.show()

def show_deimage(img):
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.axis('off')
    plt.savefig('temp/OFDM_de.png')
    #plt.show()

def show_coimage(image_path, img):
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(img, cmap='gray')
    plt.title('编解码后的图像')
    plt.axis('off')

    plt.savefig('temp/OFDM_co.png')
    #plt.show()

