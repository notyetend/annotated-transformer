import numpy as np
import tensorflow as tf

from transformer_tutorial_tf2.transformer import Transformer
from transformer_tutorial_tf2.mask import create_look_ahead_mask, create_padding_mask
from transformer_tutorial_tf2.attention import scaled_dot_product_attention

input_vocab_size = 8500
target_vocab_size = 8000
input_seq_len = 64
target_seq_len = 70
d_model = 512
max_pos_encoding = max(input_vocab_size, target_vocab_size)  # should be larger than input_seq_len
batch_size = 768

num_encoder_layers = num_decoder_layers = 2
num_head = 8
d_k = d_model // num_head
dff = 2048
rate = 0.1  # dropout


def test_create_padding_mask():
    _batch_size = 3
    _vocab_size = 5
    _input_seq_len_k = 4
    # seq = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    _input_seq = tf.constant(np.random.randint(low=0,
                                               high=_vocab_size,
                                               size=(_batch_size, _input_seq_len_k)))
    print(_input_seq)

    padding_mask = create_padding_mask(_input_seq)
    print(_input_seq, padding_mask)
    assert padding_mask.shape == tf.TensorShape([_batch_size, 1, 1, _input_seq_len_k])


def test_create_look_ahead_mask():
    _seq_len = 3  # seq_len_q == seq_len_k
    _look_ahead_mask = create_look_ahead_mask(_seq_len)
    print(_look_ahead_mask)


def test_scaled_dot_product_attention():
    input_seq = tf.constant(np.random.randint(low=0,
                                              high=input_vocab_size,
                                              size=(batch_size, input_seq_len)))

    qw = tf.random.normal(shape=(batch_size, num_head, input_seq_len, d_k))
    kw = tf.random.normal(shape=(batch_size, num_head, input_seq_len, d_k))
    vw = tf.random.normal(shape=(batch_size, num_head, input_seq_len, d_k))

    padding_mask = create_padding_mask(input_seq)
    print('shape of padding mask:', padding_mask)
    out, attn = scaled_dot_product_attention(q=qw, k=kw, v=vw, mask=padding_mask)
    print(out.shape, attn.shape)

    look_ahead_mask = create_look_ahead_mask(input_seq_len)
    print('shape of look_ahead_mask:', look_ahead_mask)
    out, attn = scaled_dot_product_attention(q=qw, k=kw, v=vw, mask=look_ahead_mask)
    print(out.shape, attn.shape)


def test_masking():
    _batch_size = 5
    _num_head = 2
    _seq_len_q = 3
    _seq_len_k = 4
    _vocab_size = 3

    input_seq = tf.constant(np.random.randint(low=0,
                                              high=_vocab_size,
                                              size=(_batch_size, _seq_len_k)))
    print(input_seq)

    scaled_attention_logits = tf.random.uniform(
        shape=(_batch_size, _num_head, _seq_len_q, _seq_len_k),
        dtype=tf.float32,
        minval=0.9,
        maxval=1.
    )
    print(scaled_attention_logits)

    padding_mask = create_padding_mask(input_seq)
    print('shape of padding mask:', padding_mask)
    scaled_attention_logits += (padding_mask * -1e9)
    print(scaled_attention_logits)

    look_ahead_mask = create_look_ahead_mask(_seq_len_q)
    print('shape of look_ahead_mask:', look_ahead_mask)


def test_transformer():
    """
    >>> out, attn = test_transformer()
    >>> out.shape
    >>> attn.keys()
    """


    transformer = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        num_heads=num_head,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_pos_encoding=max_pos_encoding,
        rate=rate
    )

    input_seq = tf.random.uniform((batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200)
    target_seq = tf.random.uniform((batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200)

    tr_out, attn_weights = transformer(
        input_seq,
        target_seq=target_seq,
        training=True,
        look_ahead_mask=None,
        enc_padding_mask=None,
        dec_padding_mask=None
    )

    assert tr_out.shape == tf.TensorShape([batch_size, target_seq_len, target_vocab_size])

    return tr_out, attn_weights
