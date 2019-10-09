# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers.merge import add
from keras.layers.recurrent import GRU
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D

__doc__ = "模型组件"


class ModelPlugIn:

    @classmethod
    def conv2d(cls, size):
        return Conv2D(size, (3, 3), use_bias=True, activation="relu",
                      padding="same", kernel_initializer="he_normal")

    @classmethod
    def norm(cls, x):
        return BatchNormalization(axis=-1)(x)

    @classmethod
    def max_pool(cls, x):
        return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

    @classmethod
    def dense(cls, units, activation="relu"):
        return Dense(units, activation=activation, use_bias=True, kernel_initializer="he_normal")

    @classmethod
    def cnn_cell(cls, size, x, pool=True):
        x = ModelPlugIn.norm(ModelPlugIn.conv2d(size)(x))
        x = ModelPlugIn.norm(ModelPlugIn.conv2d(size)(x))
        if pool:
            x = ModelPlugIn.max_pool(x)
        return x

    @classmethod
    def ctc_lambda(cls, args):
        labels, y_predict, input_length, label_length = args
        y_predict = y_predict[:, :, :]
        return K.ctc_batch_cost(labels, y_predict, input_length, label_length)

    @classmethod
    def bi_gru(cls, units, x):
        x = Dropout(0.2)(x)
        y1 = GRU(units, return_sequences=True,
                 kernel_initializer='he_normal')(x)
        y2 = GRU(units, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal')(x)
        y = add([y1, y2])
        return y
