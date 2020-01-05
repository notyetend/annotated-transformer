import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_angles(pos, i, d_model):
    """

    Parameters
    ----------
    pos : position(index of token in a sentence)
    i: dimension(idx between 0 and d_model), d_model = E for vanilla transformer_annotated setting
    d_model

    Returns
    -------

    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_pos_encoding, d_model):
    """
    >>> _d_model = 12  # i
    >>> _vocab_size = 6  # i
    >>> _angles = get_angles(np.arange(_vocab_size)[:, np.newaxis], np.arange(_d_model)[np.newaxis, :], _d_model)
    >>> _angles.shape  # _vocab_size by _d_model

    >>> _pe = positional_encoding(_vocab_size, _d_model)
    >>> _pe.shape
    """
    angle_rads = get_angles(np.arange(max_pos_encoding)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, input_vocab_size, max_pos_encoding, rate=0.1):
        """

        Parameters
        ----------
        d_model
        input_vocab_size
        max_pos_encoding: input_seq_len
        rate: dropout rate
        """
        super(TransformerEmbedding, self).__init__()

        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # ?
        x += self.pos_encoding[:, :seq_len, :]
        return self.dropout(x, training=training)


if __name__ == '__main__':
    _max_pos_encoding = 4096
    _input_seq_len = 100
    assert _max_pos_encoding >= _input_seq_len
    _d_model = 512
    _input_vocab_size = 30000
    _rate = 0.1
    _batch_size = 64

    _input_seq = tf.constant(np.random.randint(low=1, high=_input_vocab_size, size=(_batch_size, _input_seq_len)))
    """
    test plot of positional encoding
    """
    _pos_encoding = positional_encoding(50, _d_model)
    print(_pos_encoding.shape)

    plt.pcolormesh(_pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, _d_model))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

    """
    test for positional encoding
    """
    _pe = positional_encoding(max_pos_encoding=_max_pos_encoding, d_model=_d_model)
    _pe.shape  # (1, _max_pos_encoding, _d_model)
    _pe[:, :_input_seq_len, :].shape  # (1, _max_pos_encoding, d_model)

    """
    test for TransformerEmbedding
    """
    _te = TransformerEmbedding(_d_model, _input_vocab_size, _max_pos_encoding, _rate)
    _te_out = _te(_input_seq, training=True)
    _te_out.shape

