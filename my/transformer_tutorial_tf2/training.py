
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

"""
Getting dataset
"""
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

"""
Tokenize
"""
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13
)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13
)

"""
Testing tokenizer
"""
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
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2


MAX_LENGTH = 40


def filter_max_length(x, y, max_length=MAX_LENGTH):
    """
    Check if length of x and y is smaller than or equal to max_length.
    """
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def tf_encode(pt, en):
    """
    ?
    """
    return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)  # filtering sentences with it's length
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))



