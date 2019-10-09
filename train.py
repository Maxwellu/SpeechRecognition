# -*- coding: utf-8 -*-
import warnings

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback

from settings import *
from preprocess.data_parse import DataGenerator
from models.language import LanguageHyperParams
from models.acoustics import AcousticsHyperParams
from models.acoustics.cnn_ctc import AcousticsModel
from models.language.transformer import LanguageModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
root = os.path.dirname(os.path.abspath(__file__))


class EarlyStoppingByLoss(Callback):
    """
    ```监测训练集的损失, 损失小于value, 则停止训练```

    [回归]monitor: loss, val_loss
    [分类]monitor: acc
    """
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(EarlyStoppingByLoss, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR: %.2f" % (epoch, self.value))
            self.model.stop_training = True


class Train:

    def __init__(self, train_data, epoch):
        self.train_data = train_data
        self.epoch = epoch

    def train_acoustics(self):
        """声学模型训练"""
        am_args = AcousticsHyperParams.ap
        am_args.vocab_size = self.train_data.size_am_vocab
        am = AcousticsModel(am_args)
        model_h5 = os.path.join(root, "outcome/acoustics/model.h5")
        if os.path.exists(model_h5):
            am.ctc_model.load_weights(model_h5)
        steps = self.train_data.size_wav_list // BATCH_SIZE
        filename = "model_{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint_path = os.path.join(root, "outcome/acoustics/checkpoint")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        cp = os.path.join(checkpoint_path, filename)
        mc = ModelCheckpoint(cp, monitor='loss', save_weights_only=True, verbose=1, save_best_only=True)
        es = EarlyStoppingByLoss(monitor='loss', value=1, verbose=1)
        train_batch = self.train_data.get_am_batch()
        am.ctc_model.fit_generator(train_batch, steps_per_epoch=steps, epochs=self.epoch, callbacks=[mc, es])
        am.ctc_model.save_weights(model_h5)

    def train_language(self):
        """语言模型训练"""
        lm_args = LanguageHyperParams.lp
        lm_args.input_vocab_size = self.train_data.size_pny_vocab
        lm_args.label_vocab_size = self.train_data.size_han_vocab
        lm = LanguageModel(lm_args)
        batch_num = self.train_data.size_wav_list // BATCH_SIZE
        with lm.graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph=lm.graph) as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            add_num = 0
            lm_checkpoint = os.path.join(root, "outcome/language/checkpoint")
            if os.path.exists(lm_checkpoint):
                lm_path = os.path.join(root, "outcome/language")
                latest = tf.train.latest_checkpoint(lm_path)
                add_num = int(latest.split('_')[-1])
                saver.restore(sess, latest)
            tensor_board = os.path.join(root, "outcome/language/tensorboard")
            writer = tf.summary.FileWriter(tensor_board, tf.get_default_graph())
            for k in range(self.epoch):
                total_loss = 0
                batch = self.train_data.get_lm_batch()
                for i in range(batch_num):
                    input_batch, label_batch = next(batch)
                    feed = {lm.x: input_batch, lm.y: label_batch}
                    cost, _ = sess.run([lm.mean_loss, lm.train_op], feed_dict=feed)
                    total_loss += cost
                    if (k * batch_num + i) % 10 == 0:
                        rs = sess.run(merged, feed_dict=feed)
                        writer.add_summary(rs, k * batch_num + i)
                print('epoch', k + 1, ': average loss = ', total_loss / batch_num)
            lm_model = os.path.join(root, "outcome/language/model_%d" % (self.epoch + add_num))
            saver.save(sess, lm_model)
            writer.close()

    def run(self):
        self.train_acoustics()
        self.train_language()


if __name__ == '__main__':
    data_gen = DataGenerator()
    Train(data_gen, EPOCH).run()
