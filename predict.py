# -*- coding: utf-8 -*-
import os
import json

import numpy as np
import tensorflow as tf

from preprocess.audio import AudioProcess
from preprocess.data_parse import DataGenerator
from models.language import LanguageHyperParams
from models.acoustics import AcousticsHyperParams
from models.acoustics.cnn_ctc import AcousticsModel
from models.language.transformer import LanguageModel

root = os.path.dirname(os.path.abspath(__file__))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class Predict:

    def __init__(self, file):
        self.file = file

        with open("vocab_json/vocab_counter.json", "r") as f:
            data = json.load(f)
            self.am_vocab_len = data.get("am_vocab")
            self.pny_vocab_len = data.get("pny_vocab")
            self.han_vocab_len = data.get("han_vocab")

        with open("vocab_json/am_vocab.json", "r") as f:
            self.am_vocab = json.load(f)

        with open("vocab_json/pny_vocab.json", "r") as f:
            self.pny_vocab = json.load(f)

        with open("vocab_json/han_vocab.json", "r") as f:
            self.han_vocab = json.load(f)

    def load_acoustics_model(self):
        am_args = AcousticsHyperParams.ap
        am_args.vocab_size = self.am_vocab_len
        am = AcousticsModel(am_args)
        am.ctc_model.load_weights(os.path.join(root, "outcome/acoustics/model.h5"))
        return am

    def load_language_model(self):
        lm_args = LanguageHyperParams.lp
        lm_args.input_vocab_size = self.pny_vocab_len
        lm_args.label_vocab_size = self.han_vocab_len
        lm = LanguageModel(lm_args)
        sess = tf.Session(graph=lm.graph)
        with lm.graph.as_default():
            saver = tf.train.Saver()
        with sess.as_default():
            latest = tf.train.latest_checkpoint(os.path.join(root, "outcome/language"))
            saver.restore(sess, latest)
        return lm, sess

    @staticmethod
    def wav2array(file):
        wav_data_lst = []
        frequency_bank = AudioProcess.frequency_domain(file)
        pad_bank = np.zeros((frequency_bank.shape[0] // 8 * 8 + 8, frequency_bank.shape[1]))
        pad_bank[:frequency_bank.shape[0], :] = frequency_bank
        wav_data_lst.append(pad_bank)
        pad_wav_data, input_length = DataGenerator.wav_padding(wav_data_lst)
        return pad_wav_data

    def run(self):
        pad_wav_data = self.wav2array(self.file)

        am = self.load_acoustics_model()
        lm, sess = self.load_language_model()

        am_predict = am.model.predict(pad_wav_data, steps=1)
        text = AudioProcess.decoder(am_predict, self.am_vocab)
        text = ' '.join(text)
        print("AM:-)", text)

        with sess.as_default():
            text = text.strip('\n').split(' ')
            x = np.array([self.pny_vocab.index(pny) for pny in text])
            x = x.reshape(1, -1)
            lm_predict = sess.run(lm.preds, {lm.x: x})
            got = ''.join(self.han_vocab[idx] for idx in lm_predict[0])
            print("LM:-)", got)
        sess.close()

        return got


if __name__ == "__main__":
    fs = "/Users/Maxwell_Lu/Documents/" \
         "/audio_data/data_thchs30/train/A11_1.wav"
    Predict(fs).run()
