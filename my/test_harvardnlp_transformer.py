from havardnlp_transformer import *


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
    for _ in range(nbatches):
        data = torch.from_numpy(np.random.randint(low=1, high=V, size=(batch, max_words_in_sentence)))
        data[:, 0] = 1  # 1 for first column
        src = Variable(data, requires_grad=False).type(torch.long)
        tgt = Variable(data, requires_grad=False).type(torch.long)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        norm = norm.type(torch.FloatTensor)  # added by kdw
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss.data.item() * norm


def test_training():
    # Train the simple copy task.
    V = 7  # size of word dictionary
    criterion = LabelSmoothing(
        size=V,
        padding_idx=0,
        smoothing=0.0
    )

    model = make_model(
        src_vocab=V,
        tgt_vocab=V,
        N=2
    )

    model_opt = NoamOpt(
        model_size=model.src_embed[0].d_model,
        factor=1,
        warmup=400,  # lr will grow until 'warmup' steps and then shrink
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    for epoch in range(10):
        model.train()
        run_epoch(
            data_iter=data_gen(V=V, batch=5, nbatches=20, max_words_in_sentence=4),
            model=model,
            loss_compute=SimpleLossCompute(model.generator, criterion, model_opt)
        )
        model.eval()
        print(run_epoch(data_gen(V=V, batch=5, nbatches=5), model,
                        SimpleLossCompute(model.generator, criterion, None)))


# ##### params
max_words_in_sentence = 4  # of words in each sentence
batch_size = 5  # of sentences
size_dict = 7  # size of word dictionary
d_model = 12
hidden_size_pff = 11
num_head = 2
dropout_rate = 0.1
num_encoder_layer = 2
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
em = Embeddings(d_model=d_model, vocab=size_dict)
pe = PositionalEncoding(d_model=d_model, dropout=dropout_rate)
o_pe = pe(em(batch0.src))  # input to encoder, [5, 4, 12]


# encoder layer v1 ##############################
def get_encoder_output(num_head, d_model):
    """
    Encoder

    Args:
        num_head:
        d_model:

    Returns:

    Examples:
        >>> get_encoder_output(num_head, d_model)


    """
    # 1st sublayer of 1st encoder
    mha = MultiHeadedAttention(h=num_head, d_model=d_model)
    norm = LayerNorm(d_model)
    o_sub_layer1 = mha(norm(o_pe), norm(o_pe), norm(o_pe), batch0.src_mask)  # [5, 4, 12]

    # 2nd sublayer of 1st encoder
    pff = PositionwiseFeedForward(d_model=d_model, d_ff=hidden_size_pff)
    o_sub_layer2 = o_sub_layer1 + nn.Dropout(dropout_rate)(pff(norm(o_sub_layer1)))  # [5, 4, 12]
    return o_sub_layer2
##############################


# encoder layer v2 ##############################
def get_encoder_output(x, num_head, d_model, dropout_rate):
    """

    Args:
        x:
        num_head:
        d_model:

    Returns:

    Examples:
        >>> get_encoder_output(o_pe, num_head, d_model, dropout_rate)
    """
    class MySublayerConnection(nn.Module):
        def __init__(self, size, dropout_rate):
            super(MySublayerConnection, self).__init__()
            self.norm = nn.LayerNorm(size)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

    sublayers = clones(MySublayerConnection(d_model, dropout_rate), 2)

    mha = MultiHeadedAttention(h=num_head, d_model=d_model)  # 1st sublayer
    o_sub_layer1 = sublayers[0](x, lambda x: mha(x, x, x, batch0.src_mask))

    pff = PositionwiseFeedForward(d_model=d_model, d_ff=hidden_size_pff)  # 2nd sublayer
    o_sub_layer2 = sublayers[1](o_sub_layer1, pff)
    return o_sub_layer2
##############################


# encoder layer v3 ##############################
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
    >>> el.forward(o_pe, batch0.src_mask)  # [5, 4, 12]
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
        self.sublayers = clones(MySublayerConnection(size, dropout_rate), 2)
        self.mha = MultiHeadedAttention(h=num_head, d_model=size)  # 1st sublayer
        self.pff = PositionwiseFeedForward(d_model=size, d_ff=hidden_size_pff)  # 2nd sublayer

    def forward(self, x, mask):
        o_sub_layer1 = self.sublayers[0](x, lambda x: self.mha(x, x, x, mask))
        return self.sublayers[1](o_sub_layer1, self.pff)
##############################


# encoder v1 ##############################
class MyEncoder(nn.Module):
    """
    >>> el = MyEncoderLayer(d_model, dropout_rate, num_head, hidden_size_pff)
    >>> encoder = MyEncoder(el, 2)
    >>> encoder.forward(o_pe, batch0.src_mask)  # [5, 4, 12]
    """
    def __init__(self, encoder, num_encoder):
        super(MyEncoder, self).__init__()
        self.encoders = clones(encoder, num_encoder)
        self.norm = LayerNorm(encoder.size)

    def forward(self, x, mask):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return self.norm(x)

##############################


el = MyEncoderLayer(d_model, dropout_rate, num_head, hidden_size_pff)
encoder = MyEncoder(el, 2)
encoder.forward(o_pe, batch0.src_mask)  # [5, 4, 12]