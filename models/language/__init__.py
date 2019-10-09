# -*- coding: utf-8 -*-
import tensorflow as tf

__doc__ = "语言模型超参数"


class LanguageHyperParams:

    lp = tf.contrib.training.HParams(
        num_heads=8,
        num_blocks=6,
        input_vocab_size=50,
        label_vocab_size=50,
        max_length=1000,  # 每一个音频文件最多不超过max_length个汉字
        hidden_units=512,
        dropout_rate=0.2,
        lr=0.0003,
        is_training=True)
