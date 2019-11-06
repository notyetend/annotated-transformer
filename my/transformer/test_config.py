import numpy as np


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
