# -*- coding: utf-8 -*-
"""
Created at 2019-10-30

@author: dongwan.kim

Converting 'https://nlp.seas.harvard.edu/2018/04/03/attention.html'
 which is pytorch implementation
 to Keras implementation.

# ToDo: copy layer test with simple multi hidden layer regression.
"""
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense, Flatten, Conv1D, Dropout, Embedding, Input, Lambda, Layer, Softmax
)
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K


class PositionalEncodingK(Layer):
    """
    >>> max_words_in_sentence = 4  # of words in each sentence
    >>> batch_size = 5  # of sentences
    >>> size_dict = 7  # size of word dictionary
    >>> d_model = 12
    >>> hidden_size_pff = 11
    >>> num_head = 2
    >>> dropout_rate = 0.1
    >>> num_encoder_layer = 2
    >>> learning_rate = 0.001

    >>> # test implementation
    >>> pe = np.zeros([max_words_in_sentence, d_model]); print(pe, pe.shape)
    >>> position = np.expand_dims(np.array(range(max_words_in_sentence)), 1); print(position, position.shape)
    >>> div_term = np.exp(np.arange(start=0.0, stop=d_model, step=2) * -(math.log(10000.0) / d_model)); print(div_term, div_term.shape)
    >>> pe[:, 0::2] = np.sin(position * div_term)
    >>> pe[:, 1::2] = np.cos(position * div_term)
    >>> pe = np.expand_dims(pe, 0); print(pe, pe.shape)


    >>> # plotting
    >>> d_model = 12
    >>> num_sentences = 1
    >>> num_tokens_in_sentence = 100
    >>> plt.figure(figsize=(15, 5))
    >>> pe = PositionalEncodingK(d_model=d_model, dropout_rate=0)
    >>> y = pe(K.zeros((num_sentences, num_tokens_in_sentence, d_model)))
    >>> plt.plot(np.arange(num_tokens_in_sentence), K.eval(y)[0, :, 4:8])
    >>> plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    >>> plt.show()
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, **kwargs):
        """

        Parameters
        ----------
        max_len: max number of tokens in sentence.
        d_model: embedding dim
        kwargs
        """
        super(PositionalEncodingK, self).__init__(**kwargs)

        self.dropout = Dropout(rate=dropout_rate)

        pe = np.zeros([max_len, d_model])
        position = np.expand_dims(np.array(range(max_len)), 1)
        div_term = np.exp(
            np.arange(start=0.0, stop=d_model, step=2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = np.expand_dims(pe, 0)

    def call(self, x):
        x = x + K.constant(self.pe[:, :x.shape[1].value])
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class EmbeddingsK(Layer):
    """
    >>> x = K.constant([[0, 6, 1, 1, 1]]); print(x, x.shape)  # one sentence with 5 token
    >>> y = EmbeddingsK(d_model=12, vocab=7)(x)  # embedding on 12 dim for 7 tokens total.
    >>> out = K.eval(y)
    >>> print(out, out.shape)

    >>> np.random.seed(0)
    >>> emb_weight = np.random.rand(7, 12)  # total 7 tokens and hidden size is 12
    >>> x = K.constant([list(range(7))]); print(x, x.shape)  # one sentence with 5 token
    >>> y = EmbeddingsK(d_model=12, vocab=7, weight=emb_weight)(x)  # embedding on 12 dim for 7 tokens total.
    >>> test_emb_keras = K.eval(y)
    >>> print(test_emb_keras, test_emb_keras.shape)
    >>> # np.equal(test_emb_pytorch, test_emb_keras)
    >>> # np.array_equal(test_emb_pytorch, test_emb_keras)
    """

    def __init__(self, d_model, vocab, weight=None):
        """

        Parameters
        ----------
        d_model : 512 or 1024 or ..
        vocab : size of token dict
        """
        super(EmbeddingsK, self).__init__()
        self.d_model = d_model

        if weight is None:
            self.lut = Embedding(input_dim=vocab, output_dim=d_model)
        elif isinstance(weight, np.ndarray):
            self.lut = Embedding(input_dim=vocab, output_dim=d_model, weights=[weight],
                                 trainable=False)
        else:
            raise ValueError('Invalid weight')

    def call(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class LayerNormK(Layer):
    """
    btw in TF2.0, LayerNormalization functionality is provided.

    >>> ln = LayerNormK(features=12)
    >>> x = K.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]); print(x, x.shape)  # one token with d_model=12
    >>> y = K.eval(ln(x))
    >>>
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNormK, self).__init__()
        self.features = features  # d_model
        self.eps = eps
        self.a_2 = None
        self.b_2 = None

    def build(self, _):
        """
        weights are shared for all layer normalization.
            according to description of add_weight function
            'Adds a new variable to the layer, or gets an existing one; returns it'

        Parameters
        ----------
        _

        Returns
        -------

        """
        self.a_2 = self.add_weight(
            name='layer_norm_scale',
            shape=(self.features,),
            initializer='ones',
            trainable=True
        )
        self.b_2 = self.add_weight(
            name='layer_norm_bias',
            shape=(self.features,),
            initializer='zeros',
            trainable=True
        )
        return super(LayerNormK, self).build(self.features)

    def call(self, x):
        mean = K.mean(x=x, axis=-1, keepdims=True)
        std = K.std(x=x, axis=-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GeneratorK(Layer):
    """
    linear + softmax for final output layer.

    >>> ge = GeneratorK(d_model=12, vocab=7)
    >>> x = K.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]); print(x, x.shape)  # output of final layer
    >>> y = ge(x)
    >>> out = K.eval(y)
    >>> print(out, out.shape, K.eval(K.argmax(out)))
    """

    def __init__(self, d_model, vocab):
        """

        Parameters
        ----------
        d_model: hidden size
        vocab: size of token dict
        """
        super(GeneratorK, self).__init__()
        self.proj = Dense(input_shape=(d_model,), units=vocab)

    def call(self, x):
        """
        softmax followed by log is not stable,
        need to use log_softmax after upgrade to tf 2.0
        """
        return K.log(x=K.softmax(x, axis=-1))


def subsequent_mask_k(size):
    """
    Mask out subsequent positions.

    >>> subsequent_mask(3)
    tensor([
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]
            ]], dtype=torch.uint8)  # [1, 3, 3]

    This function gives mask for a sentence with 'size' words.

    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return K.equal(K.constant(subsequent_mask), 0)


class BatchK:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = K.expand_dims(K.not_equal(src, pad), axis=-2)

        if trg is not None:
            self.trg = trg[:, :-1]  # without last token of sentence
            self.trg_y = trg[:, 1:]  # without first token of sentence
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = K.sum(K.cast(K.not_equal(self.trg_y, pad), dtype='uint8'))

    @staticmethod
    def make_std_mask(trg, pad):
        trg_mask = K.expand_dims(K.not_equal(trg, pad), axis=-2)

        trg_mask = trg_mask & subsequent_mask_k(size=trg.shape.as_list()[-1])

        return trg_mask


class EncoderLayerK(Layer):
    """

    """
    def __init__(self):
        super(EncoderLayerK, self).__init__()
        # ToDo: implement


def clones_k(module, N):
    """
    >>> d = Dense(input_shape=(d_model,), units=d_model)
    >>> d_list = clones_k(d, 4)


    Parameters
    ----------
    module: layer to be copied
    N: number of copy

    Returns
    -------

    """
    # return [copy.deepcopy(module) for _ in range(N)]  # probability not working

    # reference: https://keras.io/layers/about-keras-layers/
    config = module.get_config()
    return [type(module).from_config(config) for _ in range(N)]


def attention_k(q_w_q, k_w_k, v_w_v, mask=None, dropout=None):
    pass


class MultiHeadedAttentionK(Layer):
    """

    """
    def __init__(self, h, d_model, dropout=0.1):
        """

        Parameters
        ----------
        h: number of heads
        d_model:
        """
        super(MultiHeadedAttentionK, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h  # d_k = d_v = d_model/h
        self.h = h  # number of heads
        self.linears = clones_k(Dense(input_shape=(d_model,), units=d_model))
        self.attn = None
        self.dropout = Dropout(rate=dropout)

    def forward(self, query, key, value, mask=None):
        pass
        # ToDo: Implement


if __name__ == '__test__':
    max_words_in_sentence = 4  # of words in each sentence
    batch_size = 5  # of sentences
    size_dict = 7  # size of word dictionary
    d_model = 12
    hidden_size_pff = 11
    num_head = 2
    dropout_rate = 0.1
    num_encoder_layer = 2
    learning_rate = 0.001

    # x = Input(shape=(max_words_in_sentence,))
    src = K.constant([[0, 3, 0, 2],
                      [1, 0, 3, 2],
                      [0, 0, 0, 1],
                      [1, 0, 0, 1],
                      [3, 2, 2, 1]]);
    print(src, src.shape)
    src_mask = K.constant([[[1, 1, 1, 1]],
                           [[1, 1, 1, 1]],
                           [[1, 1, 1, 1]],
                           [[1, 1, 1, 1]],
                           [[1, 1, 1, 1]]]);
    print(src_mask, src_mask.shape)
    x = EmbeddingsK(d_model=d_model, vocab=size_dict)(src)  # embedding on 12 dim for 7 tokens total.
    x = PositionalEncodingK(d_model=d_model, dropout_rate=0.)(x)
