from transformer.havardnlp_transformer import *
from transformer.my_trf_2 import *

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
    m1 = torch.from_numpy(dummy_weight).mean(-1, keepdim=True).numpy()[:]
    m2 = K.eval(K.mean(K.constant(dummy_weight), axis=-1, keepdims=True))
    assert np.array_equal(m1, m2)

    # std comparison
    """original implementation of havardnlp use unbiased std(default option for torch.tensor.std)
    as keras don't provide unbiased std, I changed torch code to use biased std."""
    s1 = torch.from_numpy(dummy_weight).std(-1, keepdim=True).numpy()[:][0, :, :]
    s2 = K.eval(K.std(K.constant(dummy_weight), axis=-1, keepdims=True))[0, :, :]
    assert not np.array_equal(pre_floor(s1), pre_floor(s2))

    s1 = torch.from_numpy(dummy_weight).std(-1, unbiased=False, keepdim=True).numpy()[:][0, :, :]
    s2 = K.eval(K.std(K.constant(dummy_weight), axis=-1, keepdims=True))[0, :, :]
    assert np.array_equal(pre_floor(s1), pre_floor(s2))

    assert out_ln_p.shape == out_ln_k.shape
    assert np.array_equal(out_ln_p, out_ln_k)



