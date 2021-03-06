# encoding:utf-8
import math
import random
import torch
import numpy as np
from collections import Counter

import operator
from utils import save_pickle


class DataLoader(object):
    def __init__(self,
                 min_freq,
                 data_path,
                 window_size,
                 skip_header,
                 negative_num,
                 vocab_size,
                 vocab_path,
                 shuffle,
                 seed,
                 sample
                 ):

        self.window_size = window_size
        self.negative_num = negative_num
        self.min_freq = min_freq
        self.shuffle = shuffle
        self.seed = seed
        self.sample = sample
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.skip_header = skip_header
        self.vocab_size = vocab_size
        self.random_s = np.random.RandomState(seed)
        self.build_examples()
        self.build_vocab()
        self.build_negative_sample_table()
        self.subsampling()

    # 分割数据
    def split_sent(self, line):
        """
        split the sentence
        Args:
            line: sentence

        Returns:
            token list
        """
        res = line.split()
        return res

    def word_to_id(self, word, vocab):
        """
        convert token into index
        Args:
            word: token, str
            vocab: dict

        Returns:
            index
        """
        return vocab[word][0] if word in vocab else vocab['<unk>'][0]

    def build_examples(self):
        """
        process the file
        Returns:

        """
        self.examples = []
        print('read data and processing')
        with open(self.data_path, 'r') as fr:
            for i, line in enumerate(fr):
                # 数据首行为列名
                if i == 0 and self.skip_header:
                    continue
                line = line.strip("\n")
                if line:
                    self.examples.append(self.split_sent(line))

    # 建立语料库
    def build_vocab(self):
        """
        Build the dictionary
        Returns:

        """
        count = Counter()
        print("build vocab")
        for words in self.examples:
            count.update(words)
        count = {k: v for k, v in count.items()}
        count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
        all_words = [(w[0], w[1]) for w in count if w[1] >= self.min_freq]
        if self.vocab_size:
            all_words = all_words[:self.vocab_size]
        all_words = all_words + [('<unk>', 0)]
        word2id = {k: (i, v) for i, (k, v) in zip(range(0, len(all_words)), all_words)}
        self.word_frequency = {tu[0]: tu[1] for word, tu in word2id.items()}
        self.vocab = {word: tu[0] for word, tu in word2id.items()}
        print(f"vocab size: {len(self.vocab)}")
        save_pickle(data=word2id, file_path=self.vocab_path)

    def build_negative_sample_table(self):
        """
        Construct the negative sample sampling probability
        Returns:

        """
        self.negative_sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.negative_sample_table += [wid] * int(c)
        self.negative_sample_table = np.array(self.negative_sample_table)

    def reserve_ratio(self, p, total):
        """
        Args:
            p: probability
            total: scale factor
        Returns:

        """
        tmp_p = (math.sqrt(p / self.sample) + 1) * self.sample / p
        if tmp_p > 1:
            tmp_p = 1
        return tmp_p * total

    def subsampling(self, total=2 ** 32):
        """
        Data sampling to reduce the occurrence of high frequency words
        Args:
            total: scale factor
        Returns:

        """
        pow_frequency = np.array(list(self.word_frequency.values()))
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        delete_int = [self.reserve_ratio(p, total=total) for p in ratio]

        self.train_examples = []
        for example in self.examples:
            words = [self.vocab[word] for word in example if
                     word in self.vocab and delete_int[self.vocab[word]] >= random.random() * total]
            if len(words) > 0:
                self.train_examples.append(words)
        del self.examples

    def get_neg_word(self, u):
        """
        negative sample
        Args:
            u: context token
        Returns:
            list of negative samples

        """
        neg_v = []
        while len(neg_v) < self.negative_num:
            n_w = np.random.choice(self.negative_sample_table, size=self.negative_num).tolist()[0]
            if n_w != u:
                neg_v.append(n_w)
        return neg_v

    def make_iter(self):
        """
        construct sample
        Returns:

        """
        for example in self.train_examples:
            if len(example) < 2:
                continue
            # dynamic window
            reduced_window = self.random_s.randint(self.window_size)
            for i, w in enumerate(example):
                words_num = len(example)
                window_start = max(0, i - self.window_size + reduced_window)
                window_end = min(words_num, i + self.window_size + 1 - reduced_window)
                # centre token
                pos_v = [example[j] for j in range(window_start, window_end) if j != i]
                # context token
                pos_u = [w] * len(pos_v)

                neg_u = [c for c in pos_v for _ in range(self.negative_num)]
                neg_v = [v for u in pos_u for v in self.get_neg_word(u)]
                yield (torch.tensor(pos_u, dtype=torch.long),
                       torch.tensor(pos_v, dtype=torch.long),
                       torch.tensor(neg_u, dtype=torch.long),
                       torch.tensor(neg_v, dtype=torch.long))

    def __len__(self):
        """
        Returns: the number of sentences

        """
        return len([w for ex in self.train_examples for w in ex if len(ex) >= 2])



