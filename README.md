# 调制解调系统

## 简介

调制解调系统是一种用于将数字信号转换为模拟信号或者将模拟信号转换为数字信号的系统。本项目使用Python实现了一个基于图形用户界面的调制解调系统，支持多种调制技术，包括ASK（Amplitude Shift Keying）、FSK（Frequency Shift Keying）、PSK（Phase Shift Keying）、QPSK（Quadrature Phase Shift Keying）、QAM（Quadrature Amplitude Modulation）和OFDM（Orthogonal Frequency Division Multiplexing）。

## 调制技术简介

#### ASK（振幅调制）

ASK是一种简单的调制技术，它通过改变信号的振幅来传输数字信息。在ASK中，两个不同的振幅代表了不同的数字位，通常是0和1。

#### FSK（频率调制）

FSK是一种调制技术，它通过改变信号的频率来传输数字信息。在FSK中，不同的频率代表了不同的数字位，通常是0和1。

#### PSK（相位调制）

PSK是一种调制技术，它通过改变信号的相位来传输数字信息。在PSK中，不同的相位代表了不同的数字位，通常是0和1。

#### QPSK（四相位调制）

QPSK是PSK的一种扩展，它将每个数字符号映射到正交的相位点上，每个相位点代表两个比特。

#### QAM（正交幅度调制）

QAM是一种调制技术，它将振幅调制和相位调制结合起来，通过同时改变信号的振幅和相位来传输数字信息。

#### OFDM（正交频分复用）

OFDM是一种调制技术，它将数据流分成多个低速数据流，并在不同的子载波上同时传输这些数据流，以提高信道利用率。


## 如何运行

1. 确保已安装Python（推荐使用Python 3.x）。

2. 安装依赖库：pip install numpy matplotlib ttkbootstrap

3. 运行主程序：python main.py

## 功能和操作

### 二进制调制解调页面（Binary）

- 输入二进制序列：在文本框中输入二进制序列，然后点击相应的按钮生成ASK、FSK或PSK调制信号，并在右侧显示调制结果。

### QPSK调制解调页面（Qpsk_button）

- 点击“Q PSK”按钮：生成QPSK调制信号。
- 点击其他按钮：分别显示QPSK调制解调过程中的不同阶段结果。

### QAM调制解调页面（Qam）

- 选择QAM方式：在下拉菜单中选择QAM调制方式，包括16QAM、64QAM、256QAM和1024QAM。
- 点击相应按钮：显示选定QAM方式下的不同阶段结果。

### OFDM调制解调页面（Ofdm）

- 点击不同按钮：分别显示OFDM系统的简介、原理概要、构建流程、测试流程、信号结构、检索、输出图像和对比等内容。

## 文件结构

- `main.py`: 主程序入口，启动用户界面。
- `S.py`: 实现ASK、FSK和PSK调制信号生成。
- `H.py`: 实现QPSK和QAM调制信号生成。
- `O.py`: 实现OFDM系统的功能和测试。

## 注意事项

- 本项目仅供学习和研究使用，不保证在所有环境下均可正常运行。
- 如果遇到问题或有改进意见，请联系项目作者。
- 维护者：Wenhao Liu
- 电子邮件：lwh20021104@gmail.com
