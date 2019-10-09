# -*- coding: utf-8 -*-
import json
from random import shuffle

import numpy as np
from tqdm import tqdm

from settings import *
from preprocess.audio import AudioProcess

__doc__ = "数据处理类"


class DataInit:

    def __init__(self):
        """
        • self.wav_list
        ['data_thchs30/train/A11_0.wav', 'data_thchs30/train/A11_10.wav', ...]

        • self.pny_list
        [['lv4', 'shi4', 'yang2', 'chun1', 'yan1', ...], ['ta1', 'jin3', 'ping2', 'yao1', 'bu4', ...], ...]

        • self.han_list
        ['绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然', '他仅凭腰部的力量在泳道上下翻腾蛹动蛇行状如海豚一直以一头的优势领先']

        • self.am_vocab
        ['lv4', 'shi4', 'yang2', 'chun1', 'yan1', 'jing3', 'da4', 'kuai4', 'wen2', ..., '_']

        • self.pny_vocab
        ['<PAD>', 'lv4', 'shi4', 'yang2', 'chun1', 'yan1', 'jing3', 'da4', 'kuai4', 'wen2', ...]

        • self.han_vocab
        ['<PAD>', '绿', '是', '阳', '春', '烟', '景', '大', '块', '文', ...]
        """
        self.wav_list = []      # 1维列表, 存放音频文件路径
        self.pny_list = []      # 2维列表, 存放拼音
        self.han_list = []      # 1维列表, 存放一条条短语
        self.am_vocab = []      # 1维列表, 所有数据构成的不重复的拼音
        self.pny_vocab = []     # 1维列表, 所有数据构成的不重复的拼音
        self.han_vocab = []     # 1维列表, 所有数据构成的不重复的汉字
        self.source_init()      # 初始化所有实例变量

    def _parse_txt(self):
        txt_list = []
        if THCHS30:
            txt_list.append('thchs.txt')
        if AISHELL:
            txt_list.append('aishell.txt')
        if PRIME:
            txt_list.append('prime.txt')
        if STCMD:
            txt_list.append('stcmd.txt')

        if DATA_LENGTH is None:
            for txt in txt_list:
                filename = os.path.join(MAPPING_DIR, txt)
                with open(filename, "r", encoding="utf8") as f:
                    data = f.readlines()
                for line in tqdm(data):
                    wav_file, pny, han = line.split('\t')
                    self.wav_list.append(wav_file)
                    self.pny_list.append(pny.split(' '))
                    self.han_list.append(han.strip('\n'))
        else:
            count = 0
            for txt in txt_list:
                filename = os.path.join(MAPPING_DIR, txt)
                with open(filename, "r", encoding="utf8") as f:
                    for line in f:
                        if count >= DATA_LENGTH:
                            return
                        wav_file, pny, han = line.split('\t')
                        self.wav_list.append(wav_file)
                        self.pny_list.append(pny.split(' '))
                        self.han_list.append(han.strip('\n'))
                        count += 1

    def source_init(self):
        self._parse_txt()
        self.am_vocab = self.mk_am_vocab(self.pny_list)
        self.pny_vocab = self.mk_lm_pny_vocab(self.pny_list)
        self.han_vocab = self.mk_lm_han_vocab(self.han_list)
        self._vocab_writer()

    def _vocab_writer(self):
        with open('vocab_json/vocab_counter.json', 'w') as fw:
            data = {"am_vocab": self.size_am_vocab,
                    "pny_vocab": self.size_pny_vocab,
                    "han_vocab": self.size_han_vocab}
            json.dump(data, fw)
        with open('vocab_json/am_vocab.json', 'w') as fw:
            json.dump(self.am_vocab, fw)
        with open('vocab_json/pny_vocab.json', 'w') as fw:
            json.dump(self.pny_vocab, fw)
        with open('vocab_json/han_vocab.json', 'w') as fw:
            json.dump(self.han_vocab, fw)

    @staticmethod
    def mk_am_vocab(data):
        """保留不重复的拼音, 列表末尾加了一个字符串`_`"""
        vocab = []
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    @staticmethod
    def mk_lm_pny_vocab(data):
        """保留不重复的拼音, 与mk_am_vocab的区别就是列表第一个位置加了字符串`<PAD>`, 末尾没有`_`"""
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    @staticmethod
    def mk_lm_han_vocab(data):
        """保留不重复的汉字, 列表第一个位置加了字符串`<PAD>`"""
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    @property
    def size_wav_list(self):
        return len(self.wav_list)

    @property
    def size_am_vocab(self):
        return len(self.am_vocab)

    @property
    def size_pny_vocab(self):
        return len(self.pny_vocab)

    @property
    def size_han_vocab(self):
        return len(self.han_vocab)


class DataGenerator(DataInit):

    def __init__(self):
        super(DataGenerator, self).__init__()

    def get_am_batch(self):
        wav_lst_length = len(self.wav_list)
        batch_num = wav_lst_length // BATCH_SIZE
        shuffle_list = list(range(wav_lst_length))
        while True:
            if SHUFFLE:
                shuffle(shuffle_list)
            for k in range(batch_num):
                wav_data_lst = []
                label_data_lst = []
                begin = k * BATCH_SIZE
                end = begin + BATCH_SIZE
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    wav = os.path.join(DATA_DIR, self.wav_list[index])
                    frequency_bank = AudioProcess.frequency_domain(wav)  # (change, 200)
                    pad_bank = np.zeros((frequency_bank.shape[0] // 8 * 8 + 8, frequency_bank.shape[1]))
                    pad_bank[:frequency_bank.shape[0], :] = frequency_bank  # (change, 200)
                    label = self.vocab_index(self.pny_list[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if pad_bank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_bank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,    # (batch_size, change, 200, 1)
                          'the_labels': pad_label_data,  # (batch_size,)
                          'input_length': input_length,  # (batch_size, change)
                          'label_length': label_length,  # (batch_size,)
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)}  # (batch_size,)
                yield inputs, outputs

    def get_lm_batch(self):
        batch_num = len(self.pny_list) // BATCH_SIZE
        for k in range(batch_num):
            begin = k * BATCH_SIZE
            end = begin + BATCH_SIZE
            input_batch = self.pny_list[begin:end]  # 拼音
            label_batch = self.han_list[begin:end]  # 汉语
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.vocab_index(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.vocab_index(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    @staticmethod
    def vocab_index(line, vocab):
        return [vocab.index(v) for v in line]

    @staticmethod
    def wav_padding(wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([_ // 8 for _ in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    @staticmethod
    def label_padding(label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    @staticmethod
    def ctc_len(label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len
