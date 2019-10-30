# -*- coding: utf-8 -*-
"""
Created at 2019-10-30

@author: dongwan.kim

"""
import numpy as np
import math

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense, Flatten, Conv1D, Dropout, Embedding, Input, Lambda, Layer, Softmax
)
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K


def get_pos_encoding_matrix(max_len, d_emb):
    """
    Examples
    -------
    >>> pos_enc = get_pos_encoding_matrix(4, 12)
    >>> pos_enc.shape  # (4, 12)

    Parameters
    ----------
    max_len
    d_emb

    Returns
    -------

    """
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc







class PositionalEncoding(Layer):
    """
    max_words_in_sentence = 4  # of words in each sentence
    batch_size = 5  # of sentences
    size_dict = 7  # size of word dictionary
    d_model = 12
    hidden_size_pff = 11
    num_head = 2
    dropout_rate = 0.1
    num_encoder_layer = 2
    learning_rate = 0.001

    pe = np.zeros([max_words_in_sentence, d_model]); print(pe, pe.shape)
    position = np.expand_dims(np.array(range(max_words_in_sentence)), 1); print(position, position.shape)
    div_term = np.exp(np.arange(start=0.0, stop=d_model, step=2) * -(math.log(10000.0) / d_model)); print(div_term, div_term.shape)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = np.expand_dims(pe, 0); print(pe, pe.shape)
    """
    def __init__(self, d_model, dropout, max_len=5000, **kwargs):
        """

        Parameters
        ----------
        max_len: max number of tokens in sentence.
        d_model: embedding dim
        kwargs
        """
        super(PositionalEncoding, self).__init__(**kwargs)

        self.dropout = dropout

        pe = np.zeros([max_len, d_model])
        position = np.expand_dims(np.array(range(max_len)), 1)
        div_term = np.exp(np.arange(start=0.0, stop=d_model, step=2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = np.expand_dims(pe, 0)

    def call(self, x):
        x = x + K.constant(self.pe[:, :x.shape[1].value])
        return self.dropout(x)

PositionalEncoding(3, None, 4)




inputs = Input(shape=(4,))
pos_emb = PosEncodingLayer(dim_dict, dim_emb)(inputs)
src_emb = Embedding(input_dim=dim_dict, output_dim=dim_emb, input_length=dim_batch)(inputs)
add_layer = Lambda(lambda x: x[0] + x[1], output_shape=lambda x: x[0])
input_emb = add_layer([src_emb, pos_emb])

inputs.shape[1].value