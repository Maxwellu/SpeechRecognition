# -*- coding: utf-8 -*-
import tensorflow as tf

__doc__ = "声学模型超参数"


class AcousticsHyperParams:

    ap = tf.contrib.training.HParams(
        vocab_size=50,
        lr=0.0008,
        gpu_nums=1,
        is_training=True)
