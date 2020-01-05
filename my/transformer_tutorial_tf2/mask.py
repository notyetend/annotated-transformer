import tensorflow as tf


def create_padding_mask(seq):
    """

    Parameters
    ----------
    seq: (batch_size, input_seq_len)

    Returns
    -------
    padding mask:
        this pad will be applied to scaled_attention_logits
        which is shape with (batch_size, 1, 1, input_seq_len_k)
        this means for all queries and heads in batch use same mask,
        so 2nd(heads) and 3rd(seq_len_q) dim is 1

    output 1 if it is padding(0), 0 otherwise
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """

    Parameters
    ----------
    size

    Returns
    -------

    """
    """
    output 1 if it should not be used for prediction, 0 otherwise

    >>> _x = tf.random.uniform((1, 3))
    >>> _x.shape
    >>> _x.numpy()
    >>> _look_ahead_mask = create_look_ahead_mask(_x.shape[1])
    >>> _look_ahead_mask.shape
    >>> _look_ahead_mask
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len_q, seq_len_k)
