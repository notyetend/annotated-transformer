import math

from tensorflow.keras.layers import (
    Dense, Flatten, Conv1D, Dropout, Embedding, Input, Lambda, Layer, Softmax
)

class Attention(Layer):

def attention_k(q_w_q, k_w_k, v_w_v, mask=None, dropout=None):
    """

    Parameters
    ----------
    q_w_q: (batch size, num heads, num tokens in sentence, d_model / d_k), (5, 2, 4, 6)
    k_w_k
    v_w_v
    mask: (5, 1, 1, 4)
    dropout: dropout layer, not dropout rate

    Returns
    -------

    """

    def masked_fill(x, mask, target_mask_val, filled_value=-1e9):
        return x * (x != target_mask_val) + (mask == target_mask_val) * filled_value

    d_k = q_w_q.shape.as_list()[-1]
    scores = K.batch_dot(q_w_q, k_w_k, axes=[3, 3]) / math.sqrt(d_k)  # (5, 2, 4, 4)
    if mask is not None:
        scores = masked_fill(scores, mask, 0, -1e9)
    p_attn = K.softmax(scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return K.batch_dot(p_attn, v_w_v, axes=[3, 2]), p_attn
