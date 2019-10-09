# -*- coding: utf-8 -*-
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.layers import Input, Reshape, Dropout, Lambda

from models.acoustics.plugin import ModelPlugIn

__doc__ = "cnn_ctc声学模型"


class AcousticsModel:

    def __init__(self, args):
        self.vocab_size = args.vocab_size
        self.gpu_nums = args.gpu_nums
        self.lr = args.lr
        self.is_training = args.is_training
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):
        self.inputs = Input(name="the_inputs", shape=(None, 200, 1))
        self.h1 = ModelPlugIn.cnn_cell(32, self.inputs)
        self.h2 = ModelPlugIn.cnn_cell(64, self.h1)
        self.h3 = ModelPlugIn.cnn_cell(128, self.h2)
        self.h4 = ModelPlugIn.cnn_cell(128, self.h3, pool=False)
        self.h5 = ModelPlugIn.cnn_cell(128, self.h4, pool=False)
        self.h6 = Reshape((-1, 3200))(self.h5)
        self.h6 = Dropout(0.2)(self.h6)
        self.h7 = ModelPlugIn.dense(256)(self.h6)
        self.h7 = Dropout(0.2)(self.h7)
        self.outputs = ModelPlugIn.dense(self.vocab_size, activation="softmax")(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    def _ctc_init(self):
        self.labels = Input(name="the_labels", shape=[None], dtype="float32")
        self.input_length = Input(name="input_length", shape=[1], dtype="int64")
        self.label_length = Input(name="label_length", shape=[1], dtype="int64")
        self.loss_out = Lambda(ModelPlugIn.ctc_lambda, output_shape=(1,), name="ctc")\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
        if self.gpu_nums > 1:
            self.ctc_model = multi_gpu_model(self.ctc_model, gpus=self.gpu_nums)
        self.ctc_model.compile(loss={"ctc": lambda y_true, output: output}, optimizer=opt)
