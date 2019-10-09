# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from scipy.fftpack import fft
from keras import backend as K
from python_speech_features import mfcc

__doc__ = "语音信号预处理及特征参数提取"


class AudioProcess:

    Time_Window = 25  # 时间窗即帧长(单位: ms)
    Move_Window = 10  # 移动窗即帧移(单位: ms)

    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    Hamming = 0.54 - 0.46 * np.cos(2 * np.pi * x / (400 - 1))  # 汉明窗

    @classmethod
    def read_audio(cls, file):
        """
        读取音频文件
        :param file: 音频文件
        :return: 音频数据, 音频采样率(单位: 赫兹)
        """
        audio_data, sample_rate = sf.read(file)
        audio_data = np.ravel(audio_data)
        return audio_data, sample_rate

    @classmethod
    def window(cls, audio_data, sample_rate):
        """
        计算循环终止的位置, 也就是最终生成的窗数
        :param audio_data: 音频数据
        :param sample_rate: 音频采样率
        :return: 窗数
        """
        return int(len(audio_data) / sample_rate * 1000 - AudioProcess.Time_Window) // AudioProcess.Move_Window + 1

    @classmethod
    def frequency_domain(cls, file):
        """
        信号时频图
        :param file: 音频文件
        :return: 特征数据
        """
        audio_data, sample_rate = cls.read_audio(file)
        win = cls.window(audio_data, sample_rate)
        feature_data = np.zeros((win, 200), dtype=np.float)  # 用于存放最终的频率特征数据
        for i in range(win):
            p_start = i * 160
            p_end = p_start + 400
            data = audio_data[p_start:p_end]
            if data.shape[0] != 400:
                lack = 400 - data.shape[0]
                lack_zero = np.zeros([lack])
                data = np.append(data, lack_zero)
            data = data * AudioProcess.Hamming  # 加窗
            data = np.abs(fft(data))
            feature_data[i] = data[0:200]  # 设置为400除以2的值取一半数据, 因为是对称的
        feature_data = np.log(feature_data + 1)
        return feature_data

    @classmethod
    def mfcc(cls, file):
        """
        MFCC特征参数提取, 削减语音信号中与识别无关的信息的影响
        :param file: 音频文件
        :return: 特征参数
        """
        audio_data, sample_rate = cls.read_audio(file)
        feat = mfcc(audio_data, samplerate=sample_rate, numcep=26)[::3]
        feat = np.transpose(feat)
        return feat

    @classmethod
    def decoder(cls, y_predict, num2word):
        """
        解码器
        :param y_predict:
        :param num2word:
        :return:
        """
        input_length = np.array([y_predict.shape[1]], dtype=np.int32)
        r = K.ctc_decode(y_predict, input_length, greedy=True, beam_width=10, top_paths=1)
        r1 = K.get_value(r[0][0])[0]
        text = [num2word[i] for i in r1]
        return text
