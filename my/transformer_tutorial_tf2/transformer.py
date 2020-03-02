# -*- coding: utf-8 -*-
"""
Created at 2020-01-03

@author: dongwan.kim

References
- https://www.tensorflow.org/tutorials/text/transformer
- https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html

Summary
- translation Portuguese to English
-

Question
- eager tensor?
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import os
import time
import numpy as np
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Getting dataset
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']


# Tokenize
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13
)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13
)


# Testing tokenizer
sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


"""

"""
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def encode(lang1, lang2):
    """
    Add a start and end token to the input and target.
    """
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2


def tf_encode(pt, en):
    """
    convert encode function to tf function so we can use this function with Dataset.map in graph mode.
    """
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


MAX_LENGTH = 40


def filter_max_length(x, y, max_length=MAX_LENGTH):
    """
    Check if length of x and y is smaller than or equal to max_length.
    """
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)  # filtering sentences with it's length

# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
type(train_dataset)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))

pt_batch, en_batch = next(iter(val_dataset))
pt_batch, en_batch


# Positional encoding
def get_angles(pos, i, d_model):
    """

    Parameters
    ----------
    pos : position(index of token in a sentence)
    i: dimension(idx between 0 and d_model), d_model = E for vanilla transformer setting
    d_model

    Returns
    -------

    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    >>> _d_model = 12  # pos
    >>> _vocab_size = 6  # i
    >>> _angles = get_angles(np.arange(_vocab_size)[:, np.newaxis], np.arange(_d_model)[np.newaxis, :], _d_model)
    >>> _angles.shape  # _vocab_size by _d_model
    >>> _pe = positional_encoding(_d_model, _vocab_size)
    >>> _pe.shape

    >>> pos_encoding = positional_encoding(50, 512)
    >>> print(pos_encoding.shape)
    >>> plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    >>> plt.xlabel('Depth')
    >>> plt.xlim((0, 512))
    >>> plt.ylabel('Position')
    >>> plt.colorbar()
    >>> plt.show()
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Masking
def create_padding_mask(seq):
    """
    >>> x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    >>> create_padding_mask(x)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    >>> x = tf.random.uniform((1, 3))
    >>> temp = create_look_ahead_mask(x.shape[1])
    >>> temp
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
