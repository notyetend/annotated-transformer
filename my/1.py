from havardnlp_transformer import *
from tensorflow.keras.layers import Conv1D


def data_gen(V, batch, nbatches, max_words_in_sentence):
    """
    Generate random data for a src-tgt copy task.

    # 5: # of sentences
    # 4: # of words in each sentence
    # 7: size of word dictionary
    np.random.randint(low=1, high=7, size=(5, 4))  # 5 by 4 matrix

    >>> gen = data_gen(7, 5, 2)
    >>> b0 = next(gen)
    >>> b0.src
    tensor([[1, 2, 3, 2],
            [1, 2, 1, 4],
            [1, 2, 4, 5],
            [1, 1, 2, 1],
            [1, 2, 5, 5]])  # [5, 4]
    >>> b0.src_mask
    tensor([[[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]]], dtype=torch.uint8)  # [5, 1, 4]
    >>> b0.trg
    tensor([[1, 2, 3],
            [1, 2, 1],
            [1, 2, 4],
            [1, 1, 2],
            [1, 2, 5]])  # [5, 3]
    >>> b0.trg_y
    tensor([[2, 3, 2],
            [2, 1, 4],
            [2, 4, 5],
            [1, 2, 1],
            [2, 5, 5]])  # [5, 3]
    >>> b0.trg_mask
    tensor([[[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]],
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]],
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]],
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]],
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]]], dtype=torch.uint8)  # [5, 3, 3]

    >>> b0.ntokens

    >>> b0.src.shape  # (5, 4)
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(low=1, high=V, size=(batch, max_words_in_sentence)))
        data[:, 0] = 1  # 1 for first column
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield BatchP(src, tgt, 0)


# ##### params
max_words_in_sentence = 4  # of words in each sentence
batch_size = 5  # of sentences
size_dict = 7  # size of word dictionary
d_model = 12
hidden_size_pff = 11
num_head = 2
dropout_rate = 0.1
# model = make_model(
#     src_vocab=size_dict,
#     tgt_vocab=size_dict,
#     N=2  # num of transformer unit
# )

# ##### data that is composed of sentence which is sequence of word index.
np.random.seed(0)
gen_batch = data_gen(V=size_dict, batch=batch_size, nbatches=2, max_words_in_sentence=max_words_in_sentence)
batch0 = next(gen_batch)
batch0.src
batch0.src_mask
batch0.trg
batch0.trg_mask

# making input to encoder
em = EmbeddingsP(d_model=d_model, vocab=size_dict)
pe = PositionalEncodingP(d_model=d_model, dropout=dropout_rate)
o_pe = pe(em(batch0.src))  # input to encoder, [5, 4, 12]


class MySublayerConnection(nn.Module):
    def __init__(self, size, dropout_rate):
        super(MySublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MyEncoderLayer(nn.Module):
    """
    >>> el = MyEncoderLayer(d_model, dropout_rate, num_head, hidden_size_pff)
    >>> el.forward(o_pe, batch0.src_mask)
    """
    def __init__(self, size, dropout_rate, num_head, hidden_size_pff):
        """

        :param size: last dim of input and output
        :param dropout_rate:
        :param num_head:
        :param hidden_size_pff:
        """
        super(MyEncoderLayer, self).__init__()
        self.size = size
        self.sublayers = clones_p(MySublayerConnection(size, dropout_rate), 2)
        self.mha = MultiHeadedAttentionP(h=num_head, d_model=size)  # 1st sublayer
        self.pff = PositionwiseFeedForwardP(d_model=size, d_ff=hidden_size_pff)  # 2nd sublayer

    def forward(self, x, mask):
        o_sub_layer1 = self.sublayers[0](x, lambda x: self.mha(x, x, x, mask))
        return self.sublayers[1](o_sub_layer1, self.pff)
