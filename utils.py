#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random
import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    # vocabulary=[vocabulary[i:i+num_steps] for i in range(len(vocabulary)-num_steps+1)]
    # x=vocabulary[:-1]
    # y=vocabulary[1:]
    # dy=len(x)%batch_size
    # if dy!=0 :
    #     x = np.reshape(x[:-dy], [-1, batch_size, num_steps])
    #     y = np.reshape(y[:-dy], [-1, batch_size, num_steps])
    # else:
    #     x = np.reshape(x, [-1, batch_size, num_steps])
    #     y = np.reshape(y, [-1, batch_size, num_steps])
    #
    # return zip(x,y)
    ##################
    raw_x = vocabulary[:-1]
    raw_y = vocabulary[1:]
    data_partition_size = len(raw_x) // batch_size
    data_x = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    data_y = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[data_partition_size * i:data_partition_size * (i + 1)]
        data_y[i] = raw_y[data_partition_size * i:data_partition_size * (i + 1)]
    epoch_size = data_partition_size // num_steps
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
