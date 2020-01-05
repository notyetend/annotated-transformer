import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    """
    feed-forward for last dim of input.

    >>> sample_ffn = point_wise_feed_forward_network(d_model=512, dff=2048)
    >>> sample_ffn(tf.random.uniform((64, 50, 512))).shape
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
