from tensorflow.python.keras.layers import BatchNormalization
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense, Flatten, Conv1D, Dropout, Embedding, Input, Lambda, Layer, Softmax
)
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from havardnlp_transformer import *
from my_trf_2 import *


max_words_in_sentence = 4  # of words in each sentence
batch_size = 5  # of sentences
size_dict = 7  # size of word dictionary
d_model = 12
hidden_size_pff = 11
num_head = 2
dropout_rate = 0.1
num_encoder_layer = 2
learning_rate = 0.001

src_np = [[0, 3, 0, 2],
          [1, 0, 3, 2],
          [0, 0, 0, 1],
          [1, 0, 0, 1],
          [3, 2, 2, 1]]

src_mask_np = [[[1, 1, 1, 1]],
               [[1, 1, 1, 1]],
               [[1, 1, 1, 1]],
               [[1, 1, 1, 1]],
               [[1, 1, 1, 1]]]


def pre_floor(a, precision=5):
    """ floor with precision """
    return np.round(a - 0.5 * 10**(-precision), precision)


def test_positional_encoding():
    """
    >>> test_positional_encoding()

    Returns
    -------

    """
    dummy_emb_np = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')

    # pytorch
    pe_p = PositionalEncodingP(d_model=d_model, dropout=0.)
    out_pe_p = pe_p.forward(torch.from_numpy(dummy_emb_np))
    out_pe_p = out_pe_p.numpy()[:]
    out_pe_p = pre_floor(out_pe_p)

    # keras
    pe_k = PositionalEncodingK(d_model=d_model, dropout_rate=0.)
    out_pe_k = pe_k(K.constant(dummy_emb_np))
    out_pe_k = K.eval(out_pe_k)
    out_pe_k = pre_floor(out_pe_k)

    print(np.equal(out_pe_p, out_pe_k), np.array_equal(out_pe_p, out_pe_k))

    assert np.array_equal(out_pe_p, out_pe_k)


def test_embeddings():
    """

    Returns
    -------

    """
    dummy_weight = np.random.rand(size_dict, d_model).astype('float32')
    dummy_batch = np.random.randint(low=0, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))

    # pytorch
    emb_p = EmbeddingsP(d_model=d_model, vocab=size_dict, weight=dummy_weight)
    out_emb_p = emb_p(torch.from_numpy(dummy_batch).to(torch.int64))
    """
    w/o '.to(torch.int64)' you will get below error.
        RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.IntTensor instead 
        (while checking arguments for embedding)
    """

    # keras
    emb_k = EmbeddingsK(d_model=d_model, vocab=size_dict, weight=dummy_weight)
    out_emb_k = emb_k(K.constant(dummy_batch))
    out_emb_k = K.eval(out_emb_k)

    assert np.array_equal(out_emb_p, out_emb_k)


def test_layer_norm():
    dummy_weight = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')

    # pytorch
    ln_p = LayerNormP(features=d_model)
    out_ln_p = ln_p(torch.from_numpy(dummy_weight))
    out_ln_p = pre_floor(out_ln_p.detach().numpy()[:])
    out_ln_p

    # keras
    ln_k = LayerNormK(features=d_model)
    out_ln_k = ln_k(K.constant(dummy_weight))
    out_ln_k = K.eval(out_ln_k)
    out_ln_k = pre_floor(out_ln_k)
    out_ln_k

    # mean comparison
    mean_p_1 = torch.from_numpy(dummy_weight).mean(-1, keepdim=True).numpy()[:]
    mean_k_1 = K.mean(K.constant(dummy_weight), axis=-1, keepdims=True)
    assert np.array_equal(pre_floor(mean_p_1), pre_floor(K.eval(mean_k_1)))

    # std comparison
    """original implementation of havardnlp use unbiased std(default option for torch.tensor.std)
    as keras don't provide unbiased std, I changed torch code to use biased std."""
    std_p_1 = torch.from_numpy(dummy_weight).std(-1, keepdim=True).numpy()[:][0, :, :]  # unbiased
    std_k_1 = K.eval(K.std(K.constant(dummy_weight), axis=-1, keepdims=True))[0, :, :]  # biased
    assert not np.array_equal(pre_floor(std_p_1), pre_floor(std_k_1))

    std_k_2 = K.eval(K.sqrt(
        K.mean(
            K.square(
                K.constant(dummy_weight) - mean_k_1
            ),
            axis=-1,
            keepdims=True
        )
    ))[0, :, :]  # biased std
    assert np.array_equal(std_k_1, std_k_2)

    std_k_3 = K.eval(
        K.sqrt(
            K.sum(K.square(K.constant(dummy_weight) - mean_k_1), axis=-1, keepdims=True) / (d_model - 1)
        )
    )[0, :, :]  # unbiased std
    assert np.array_equal(std_p_1, std_k_3)

    std_p_2 = torch.from_numpy(dummy_weight).std(-1, unbiased=False, keepdim=True).numpy()[:][0, :, :]  # biased
    assert np.array_equal(pre_floor(std_p_2), pre_floor(std_k_1))

    assert out_ln_p.shape == out_ln_k.shape
    assert np.array_equal(out_ln_p, out_ln_k)


def test_generator():
    dummy_weight = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')

    # pytorch
    ge_p = GeneratorP(d_model=d_model, vocab=size_dict)
    out_ge_p = ge_p(torch.from_numpy(dummy_weight))
    out_ge_p = out_ge_p.detach().numpy()

    # keras
    ge_k = GeneratorK(d_model=d_model, vocab=size_dict)
    out_ge_p = ge_k(K.constant(dummy_weight))
    out_ge_p = K.eval(out_ge_p)

    print(out_ge_p[0, :, :])
    print(out_ge_p.shape, out_ge_p.shape)
    assert np.array_equal(out_ge_p, out_ge_p)


def test_linear():
    np.random.seed(0)
    num_in_node = d_model
    num_out_node = 30
    dummy_bias = np.zeros(num_out_node,)
    dummy_weight = np.random.rand(num_out_node, num_in_node).astype('float32')
    dummy_input = np.random.rand(batch_size, max_words_in_sentence, num_in_node).astype('float32')

    # keras
    input_k = Input(shape=(num_in_node,), batch_size=batch_size)
    linear_k = Dense(input_shape=(num_in_node,), units=num_out_node)
    linear_k(input_k)  # keras layer initialize weights after taking input
    linear_k.set_weights([dummy_weight.transpose(), dummy_bias])
    # keras takes weight matrix with shape of [num_in_node, num_out_node], so doing transpose
    assert np.array_equal(dummy_weight.transpose(), linear_k.get_weights()[0])  # get_weights() gives [weight, bias]
    assert np.array_equal(dummy_bias, linear_k.get_weights()[1])
    out_linear_k = K.eval(linear_k(K.constant(dummy_input)))

    # pytorch
    linear_p = nn.Linear(in_features=d_model, out_features=d_model)
    # pytorch takes weight matrix with shape of [num_out_node, num_in_node]
    linear_p.weight.data = torch.from_numpy(dummy_weight).float()
    linear_p.bias.data = torch.from_numpy(dummy_bias).float()

    assert np.array_equal(dummy_weight, linear_p.weight.detach().numpy())
    assert np.array_equal(dummy_bias, linear_p.bias.detach().numpy())
    out_linear_p = linear_p(torch.from_numpy(dummy_input)).detach().numpy()

    out_linear_k[0, 0]
    out_linear_p[0, 0]
    assert np.array_equal(pre_floor(out_linear_k, precision=3), pre_floor(out_linear_p, precision=3))
    # no idea why they are differ in bigger precision


def test_subsequent_mask():
    # pytorch
    msk_p = subsequent_mask_p(3).numpy()

    # keras
    msk_k = K.eval(subsequent_mask_k(3))
    msk_k

    assert np.array_equal(msk_p, msk_k)


def test_make_std_mask():
    dummy_batch = np.random.randint(low=1, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
    dummy_batch[:, 0] = 1
    pad = 0

    # pytorch
    trg = torch.from_numpy(dummy_batch)[:, :-1]
    std_mask_p = BatchP.make_std_mask(trg, pad)
    std_mask_p = std_mask_p.numpy()

    # keras
    trg = K.constant(dummy_batch)[:, :-1]
    std_mask_k = BatchK.make_std_mask(trg, pad)
    std_mask_k = K.eval(std_mask_k)
    std_mask_k

    assert np.array_equal(std_mask_p, std_mask_k)


def test_batch():
    dummy_batch = np.random.randint(low=1, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
    dummy_batch[:, 0] = 1
    dummy_batch = dummy_batch.astype('uint8')
    pad = 0

    # pytorch
    src = torch.from_numpy(dummy_batch)
    trg = torch.from_numpy(dummy_batch)
    ba_p = BatchP(src=src, trg=trg, pad=pad)

    # keras
    src = K.constant(dummy_batch, dtype='uint8')
    trg = K.constant(dummy_batch, dtype='uint8')
    ba_k = BatchK(src=src, trg=trg, pad=pad)

    # test src
    assert np.array_equal(ba_p.src.numpy(), K.eval(ba_k.src))

    # test trg
    assert np.array_equal(ba_p.trg.numpy(), K.eval(ba_k.trg))

    # test src mask
    assert np.array_equal(ba_p.src_mask.numpy(), K.eval(ba_k.src_mask))

    # test trg mask
    assert np.array_equal(ba_p.trg_mask.numpy(), K.eval(ba_k.trg_mask))


def test_attention():
    assert d_model % num_head == 0

    d_k = d_v = d_model // num_head  # dim of query and key is d_k
    dummy_q_w_q = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k)
    print(dummy_q_w_q.shape)
    dummy_k_w_k = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k)
    dummy_v_w_v = np.random.rand(batch_size, num_head, max_words_in_sentence, d_v)

    dummy_src_mask = np.ones(shape=(batch_size, 1, max_words_in_sentence), dtype='uint8')



    dummy_weight = np.random.rand(size_dict, d_model).astype('float32')
    dummy_batch = np.random.randint(low=0, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))

    # pytorch
    emb_p = EmbeddingsP(d_model=d_model, vocab=size_dict, weight=dummy_weight)
    out_emb_p = emb_p(torch.from_numpy(dummy_batch).to(torch.int64))
    """
    w/o '.to(torch.int64)' you will get below error.
        RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.IntTensor instead 
        (while checking arguments for embedding)
    """

    # keras
    emb_k = EmbeddingsK(d_model=d_model, vocab=size_dict, weight=dummy_weight)
    out_emb_k = emb_k(K.constant(dummy_batch))
    out_emb_k = K.eval(out_emb_k)

    # ToDo: implement further

    # test implementation
    q_w_q = K.constant(dummy_q_w_q)
    d_k = q_w_q.shape.as_list()[-1]
    scores = K.batch_dot(K.constant(dummy_q_w_q), K.constant(dummy_k_w_k), axes=[3, 3]) / math.sqrt(d_k)
    scores = K.eval(scores)


def test_masked_fill():
    assert d_model % num_head == 0

    d_k = d_v = d_model // num_head  # dim of query and key is d_k
    dummy_q_w_q = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k)
    print(dummy_q_w_q.shape)
    dummy_k_w_k = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k)
    dummy_v_w_v = np.random.rand(batch_size, num_head, max_words_in_sentence, d_v)

    # pytorch
    q_w_q = torch.from_numpy(dummy_q_w_q)
    k_w_k = torch.from_numpy(dummy_k_w_k)
    d_k = q_w_q.size(-1)
    scores = torch.matmul(q_w_q, k_w_k.transpose(-2, -1)) / math.sqrt(d_k)
    scores
    scores.shape

    dummy_batch = np.random.randint(low=1, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
    dummy_batch[:, 0] = 1
    dummy_batch = dummy_batch.astype('uint8')
    pad = 0

    # pytorch
    src = torch.from_numpy(dummy_batch)
    trg = torch.from_numpy(dummy_batch)
    ba_p = BatchP(src=src, trg=trg, pad=pad)
    src_mask = ba_p.src_mask
    src_mask
    src_mask.shape
    src_mask = src_mask.unsqueeze(1)
    src_mask.shape

    scores.masked_fill(src_mask == 0, -1e9)


def test_multi_head_attention():

    # pytorch
    mha_p = MultiHeadedAttentionP(h=2, d_model=d_model, dropout=0.)


    # keras
    mha_k = MultiHeadedAttentionK(h=2, d_model=d_model, dropout=0.)
    # ToDo: implement further

