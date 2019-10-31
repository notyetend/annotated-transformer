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
