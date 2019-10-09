# -*- coding: utf-8 -*-
import tensorflow as tf

__doc__ = "transformer语言模型"


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta
    return outputs


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True, scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs


def multi_head_attention(emb, queries, keys, num_units=None, num_heads=8, dropout_rate=0,
                        is_training=True, causality=False, scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        query_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        outputs += queries
        outputs = normalize(outputs)  # (N, T_q, C)
    return outputs


def feed_forward(inputs, num_units=(2048, 512), scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs += inputs
        outputs = normalize(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    k = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / k)


class LanguageModel:
    def __init__(self, arg):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = arg.is_training
            self.hidden_units = arg.hidden_units
            self.input_vocab_size = arg.input_vocab_size
            self.label_vocab_size = arg.label_vocab_size
            self.num_heads = arg.num_heads
            self.num_blocks = arg.num_blocks
            self.max_length = arg.max_length
            self.lr = arg.lr
            self.dropout_rate = arg.dropout_rate
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.int32, shape=(None, None))
            self.emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True,
                                 scope="enc_embed")
            self.enc = self.emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="enc_pe")
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    self.enc = multi_head_attention(emb=self.emb,
                                                    queries=self.enc,
                                                    keys=self.enc,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=False)
            self.outputs = feed_forward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])
            self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
            if self.is_training:
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
