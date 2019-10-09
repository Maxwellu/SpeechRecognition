# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf

from settings import HyperParams
from utils.util_data import GetData
from train import training_data
from utils.audio import AudioManager
from models.language import LanguageHyperParams
from models.acoustics import AcousticsHyperParams
from models.acoustics.cnn_ctc import AcousticsModel
from models.language.transformer import LanguageModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
root = os.path.dirname(os.path.abspath(__file__))


def load_acoustics_model(train_data):
    """加载声学模型"""
    am_args = AcousticsHyperParams.ap
    am_args.vocab_size = len(train_data.am_vocab)
    am = AcousticsModel(am_args)
    am.ctc_model.load_weights(os.path.join(root, "outcome/acoustics/model.h5"))
    return am


def load_language_model(train_data):
    """加载语言模型"""
    lm_args = LanguageHyperParams.lp
    lm_args.input_vocab_size = len(train_data.pny_vocab)
    lm_args.label_vocab_size = len(train_data.han_vocab)
    lm = LanguageModel(lm_args)
    sess = tf.Session(graph=lm.graph)
    with lm.graph.as_default():
        saver = tf.train.Saver()
    with sess.as_default():
        latest = tf.train.latest_checkpoint(os.path.join(root, "outcome/language"))
        saver.restore(sess, latest)
    return lm, sess


def test_data():
    """测试数据"""
    data_args = HyperParams.hp
    data_args.data_type = "test"
    data_args.data_path = os.path.join(root, "data/")
    data_args.thchs30 = True
    data_args.aishell = False
    data_args.prime = False
    data_args.stcmd = False
    data_args.batch_size = 1
    data_args.data_length = 1  # None: 使用全部数据
    data_args.shuffle = False
    return GetData(data_args)


def test_model():
    """测试训练模型"""
    data = test_data()
    am_batch = data.get_am_batch()  # 音频数据生成器

    train_data = training_data()

    am = load_acoustics_model(train_data)
    lm, sess = load_language_model(train_data)

    inputs, outputs = next(am_batch)
    x = inputs['the_inputs']
    y = data.pny_lst[0]
    am_predict = am.model.predict(x, steps=1)
    _, text = AudioManager.decode_ctc(am_predict, train_data.am_vocab)  # 将数字结果转化为文本结果
    text = ' '.join(text)
    print('原文拼音: ', ' '.join(y))
    print('识别拼音: ', text)

    with sess.as_default():
        text = text.strip('\n').split(' ')
        x = np.array([train_data.pny_vocab.index(pny) for pny in text])
        x = x.reshape(1, -1)
        lm_predict = sess.run(lm.preds, {lm.x: x})
        label = data.han_lst[0]
        got = ''.join(train_data.han_vocab[idx] for idx in lm_predict[0])
        print('原文汉字: ', label)
        print('识别汉字: ', got)
    sess.close()


if __name__ == "__main__":
    test_model()
