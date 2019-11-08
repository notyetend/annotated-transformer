import numpy as np
import torch
import torch.nn as nn


max_words_in_sentence = 4  # of words in each sentence
batch_size = 5  # of sentences
size_dict = 7  # size of word dictionary
d_model = 12
hidden_size_pff = 11
num_head = 2
dropout_rate = 0.1
num_encoder_layer = 2
learning_rate = 0.001

src_np = [[0, 3, 0, 2],
          [1, 0, 3, 2],
          [0, 0, 0, 1],
          [1, 0, 0, 1],
          [3, 2, 2, 1]]

src_mask_np = [[[1, 1, 1, 1]],
               [[1, 1, 1, 1]],
               [[1, 1, 1, 1]],
               [[1, 1, 1, 1]],
               [[1, 1, 1, 1]]]


def pre_floor(a, precision=5):
    """ floor with precision """
    return np.round(a - 0.5 * 10**(-precision), precision)


dummy_batch = np.random.randint(low=0, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
x = torch.from_numpy(dummy_batch)
x
x.shape

y = (x > 0)
y
y.shape


y = (x > 0).unsqueeze(1)
y
y.shape


y = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
y
y.shape


y = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
y
y.shape

