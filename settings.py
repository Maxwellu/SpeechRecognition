# -*- coding: utf-8 -*-
import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 音频汉字映射文件存放目录
MAPPING_DIR = os.path.join(ROOT_DIR, "mapping")

# 音频文件根目录
DATA_DIR = "/Users/Maxwell_Lu/Documents/audio_data"

# 是否使用数据集
THCHS30 = True
AISHELL = False
PRIME = False
STCMD = False

# 每个批次训练几个数据
BATCH_SIZE = 1

# 训练轮数
EPOCH = 20

# 使用数据量, None代表所有数据
DATA_LENGTH = 2

# 是否打乱数据
SHUFFLE = True
