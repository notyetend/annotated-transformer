import tensorflow as tf

from transformer_tutorial_tf2.attention import MultiHeadAttention
from transformer_tutorial_tf2.pointwise_ffn import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


if __name__ == '__main__':
    _d_model = 512
    _num_head = 8
    _dff = 2048
    _input_seq_len = 43
    _batch_size = 64

    _sample_encoder_layer = EncoderLayer(
        d_model=_d_model,
        num_heads=_num_head,
        dff=_dff
    )

    _sample_encoder_layer_output = _sample_encoder_layer(
        tf.random.uniform((_batch_size, _input_seq_len, _d_model)),
        training=False,
        mask=None
    )

    print(_sample_encoder_layer_output.shape)
