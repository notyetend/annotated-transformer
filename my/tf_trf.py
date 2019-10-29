import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Dropout, Embedding, Input, Lambda, Layer, Softmax
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def self_attn(qkv):
    query, key, value = qkv
    k_transposed = K.permute_dimensions(key, (0, 2, 1))  # [None, 128, 64]
    qk = K.batch_dot(query, k_transposed)  # [None, 64, 64]
    d_k = query.get_shape()[-1].value
    attention = Softmax(axis=-1)(qk / np.sqrt(d_k))
    attention = Dropout(rate=0.1)(attention)  # [None, 64, 64]
    o = K.batch_dot(attention, value)  # [None, 64, 128]

    o = Conv1D(filters=32, kernel_size=1, activation='relu', padding='valid')(o)  # [None, 64, 32]
    o = Dropout(rate=0.1)(o)
    return attention, o


class LayerNormalization(Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    "Unlike batch normalization, layer normalization performs exactly
    the same computation at training and test times."
    """

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class PosEncodingLayer:
    def __init__(self, max_len, d_emb):
        self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
                                        weights=[get_pos_encoding_matrix(max_len, d_emb)])

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def __call__(self, seq, pos_input=True):
        x = seq
        if not pos_input: x = Lambda(self.get_pos_seq)(x)
        return self.pos_emb_matrix(x)


dim_batch = 50  # sentence length
dim_dict = 1000
dim_emb = 32
learning_rate = 0.001

inputs = Input(shape=(dim_batch,))
pos_emb = PosEncodingLayer(dim_dict, dim_emb)(inputs)
src_emb = Embedding(input_dim=dim_dict, output_dim=dim_emb, input_length=dim_batch)(inputs)
add_layer = Lambda(lambda x: x[0] + x[1], output_shape=lambda x: x[0])
input_emb = add_layer([src_emb, pos_emb])

# q = BatchNormalization(**bn_param)(input_emb)
q = LayerNormalization()(input_emb)
q = Conv1D(filters=16, kernel_size=1, activation='relu', padding='valid')(q)  # [None, 64, 128]
q = Dropout(rate=0.1)(q)

# k = BatchNormalization(**bn_param)(input_emb)
k = LayerNormalization()(input_emb)
k = Conv1D(filters=16, kernel_size=1, activation='relu', padding='valid')(k)  # [None, 64, 128]
k = Dropout(rate=0.1)(k)

# v = BatchNormalization(**bn_param)(input_emb)
v = LayerNormalization()(input_emb)
v = Conv1D(filters=16, kernel_size=1, activation='relu', padding='valid')(v)  # [None, 64, 128]
v = Dropout(rate=0.1)(v)

attn, x = Lambda(self_attn)([q, k, v])

flat = Flatten()(x)
dense = Dense(512, activation='relu')(flat)
y = Dense(1, activation='sigmoid')(dense)

model = Model(inputs, y)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
