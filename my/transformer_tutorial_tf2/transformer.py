import tensorflow as tf

from transformer_tutorial_tf2.decoder import DecoderLayer
from transformer_tutorial_tf2.embedding import TransformerEmbedding
from transformer_tutorial_tf2.encoder import EncoderLayer


class Transformer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            target_vocab_size,
            max_pos_encoding,
            rate=0.1
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_enc_layers = num_encoder_layers
        self.num_dec_layers = num_decoder_layers

        self.input_embedding = TransformerEmbedding(d_model, input_vocab_size, max_pos_encoding, rate)
        self.target_embedding = TransformerEmbedding(d_model, target_vocab_size, max_pos_encoding, rate)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_encoder_layers)]
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_decoder_layers)]

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_seq, target_seq, training, look_ahead_mask, enc_padding_mask, dec_padding_mask):
        x_enc = self.input_embedding(input_seq, training)
        for i in range(self.num_enc_layers):
            x_enc = self.enc_layers[i](x_enc, training, enc_padding_mask)

        attention_weights = {}
        x_dec = self.target_embedding(target_seq, training)
        for i in range(self.num_dec_layers):
            x_dec, attn1, attn2 = self.dec_layers[i](x_dec, x_enc, training,
                                                     look_ahead_mask, dec_padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = attn1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = attn2

        return self.final_layer(x_dec), attention_weights
