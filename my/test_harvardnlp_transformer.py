from havardnlp_transformer import *


def data_gen(V, batch, nbatches, max_words_in_sentence):
    """
    Generate random data for a src-tgt copy task.

    # 5: # of sentences per batch == batch(2nd arg)
    # 4: # of words in each sentence
    # 7: size of word dictionary
    np.random.randint(low=1, high=7, size=(5, 4))  # 5 by 4 matrix

    >>> gen = data_gen(7, 5, 2, 4)
    >>> b0 = next(gen)
    >>> b0.src
    >>> b0.trg
    >>> b0.src.shape  # [5, 4]
    >>> b0.ntokens  # 15
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

    >>> b0.ntokens  # 15

    >>> b0.src.shape  # (5, 4)
    """
    for _ in range(nbatches):
        data = torch.from_numpy(np.random.randint(low=1, high=V, size=(batch, max_words_in_sentence)))
        data[:, 0] = 1  # 1 for first column
        src = Variable(data, requires_grad=False).type(torch.long)
        tgt = Variable(data, requires_grad=False).type(torch.long)
        yield BatchP(src, tgt, 0)


def first_example():
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

    # Train the simple copy task.
    V = 7  # size of word dictionary
    criterion = LabelSmoothingP(size=V, padding_idx=0, smoothing=0.0)
    model = make_model_p(src_vocab=V, tgt_vocab=V, N=2)

    model_opt = NoamOptP(
        model_size=model.src_embed[0].d_model,
        factor=1,
        warmup=400,  # lr will grow until 'warmup' steps and then shrink
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    for epoch in range(10):
        model.train()
        run_epoch_p(
            data_iter=data_gen(V=V, batch=5, nbatches=5, max_words_in_sentence=4),
            model=model,
            loss_compute=SimpleLossCompute(model.generator, criterion, model_opt)
        )
        model.eval()
        print(run_epoch_p(data_gen(V=V, batch=5, nbatches=5, max_words_in_sentence=4), model,
                          SimpleLossCompute(model.generator, criterion, None)))


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
    mha = MultiHeadedAttentionP(h=num_head, d_model=d_model)
    norm = LayerNormP(d_model)
    o_sub_layer1 = mha(norm(o_pe), norm(o_pe), norm(o_pe), batch0.src_mask)  # [5, 4, 12]

    # 2nd sublayer of 1st encoder
    pff = PositionwiseFeedForwardP(d_model=d_model, d_ff=hidden_size_pff)
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

    sublayers = clones_p(MySublayerConnection(d_model, dropout_rate), 2)

    mha = MultiHeadedAttentionP(h=num_head, d_model=d_model)  # 1st sublayer
    o_sub_layer1 = sublayers[0](x, lambda x: mha(x, x, x, batch0.src_mask))

    pff = PositionwiseFeedForwardP(d_model=d_model, d_ff=hidden_size_pff)  # 2nd sublayer
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
        self.sublayers = clones_p(MySublayerConnection(size, dropout_rate), 2)
        self.mha = MultiHeadedAttentionP(h=num_head, d_model=size)  # 1st sublayer
        self.pff = PositionwiseFeedForwardP(d_model=size, d_ff=hidden_size_pff)  # 2nd sublayer

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
        self.encoders = clones_p(encoder, num_encoder)
        self.norm = LayerNormP(encoder.size)

    def forward(self, x, mask):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return self.norm(x)


##############################
test = False
if test:
    # data for decoder
    batch0.trg
    batch0.trg_mask
    el = MyEncoderLayer(d_model, dropout_rate, num_head, hidden_size_pff)
    encoder = MyEncoder(el, 6)
    o_encoder = encoder.forward(o_pe, batch0.src_mask)  # [5, 4, 12]

    # making input to encoder
    em = EmbeddingsP(d_model=d_model, vocab=size_dict)
    pe = PositionalEncodingP(d_model=d_model, dropout=dropout_rate)
    o_pe_decoder = pe(em(batch0.trg))  # input to encoder, [5, 4, 12]

    # decoder layer v1 ##############################
    self_attn = MultiHeadedAttentionP(h=num_head, d_model=d_model)
    src_attn = MultiHeadedAttentionP(h=num_head, d_model=d_model)
    pff = PositionwiseFeedForwardP(d_model=d_model, d_ff=hidden_size_pff)

    # first input: output of embedding and positional encoding for decoder
    # second input: output of encoder this is used as key and value,
    #   output of previous sublayer this is used as query
    sublayers = clones_p(MySublayerConnection(d_model, dropout_rate), 3)
    o_sublayer0 = sublayers[0](o_pe_decoder, lambda x: self_attn(
        query=o_pe_decoder,
        key=o_pe_decoder,
        value=o_pe_decoder,
        mask=batch0.trg_mask))
    o_sublayer1 = sublayers[1](o_sublayer0, lambda x: src_attn(
        query=o_sublayer0,
        key=o_encoder,
        value=o_encoder,
        mask=batch0.src_mask))  # why use src_mask, not trg_mask???
    # -> 이미 이런 단계(sublayers[0])에서 타겟 문장의 정보를 필터링 했으므로, 추가로 필터링 할 필요가 없어 보인다.
    o_sublayer2 = sublayers[2](o_sublayer1, pff)
    o_sublayer2.shape


##############################


# decoder layer v2  ##############################
class MyDecoderLayer(nn.Module):
    """
    >>> dl = MyDecoderLayer(d_model, num_head)
    >>> o_decoder_layer = dl.forward(o_pe_decoder=o_pe_decoder, src_mask=batch0.src_mask, trg_mask=batch0.trg_mask)
    >>> o_decoder_layer.shape
    """

    def __init__(self, size, num_head):
        super(MyDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadedAttentionP(h=num_head, d_model=size)
        self.src_attn = MultiHeadedAttentionP(h=num_head, d_model=size)
        self.pff = PositionwiseFeedForwardP(d_model=size, d_ff=hidden_size_pff)
        self.sublayers = clones_p(MySublayerConnection(size, dropout_rate), 3)

    def forward(self, o_prev_decoder, o_encoder, src_mask, trg_mask):
        """
        o_prev_decoder could be output of embedding(+pe)
            or output of previous decoder
        """
        o_sublayer0 = self.sublayers[0](o_prev_decoder, lambda x: self.self_attn(
            query=o_prev_decoder,
            key=o_prev_decoder,
            value=o_prev_decoder,
            mask=trg_mask))
        o_sublayer1 = self.sublayers[1](o_sublayer0, lambda x: self.src_attn(
            query=o_sublayer0,
            key=o_encoder,
            value=o_encoder,
            mask=src_mask))  # why use src_mask, not trg_mask???
        # -> 이미 이런 단계(sublayers[0])에서 타겟 문장의 정보를 필터링 했으므로, 추가로 필터링 할 필요가 없어 보인다.
        return sublayers[2](o_sublayer1, self.pff)


##############################


# decoder
class MyDecoder(nn.Module):
    """
    >>> el = MyEncoderLayer(d_model, dropout_rate, num_head, hidden_size_pff)
    >>> encoder = MyEncoder(el, 6)
    >>> o_encoder = encoder.forward(o_pe_encoder, batch0.src_mask)  # [5, 4, 12]

    >>> decoder_layer = MyDecoderLayer(size=d_model, num_head=num_head)
    >>> o_decoder = decoder_layer.forward(o_pe_decoder, o_encoder, batch0.src_mask, batch0.trg_mask)
    >>> o_decoder.shape  # [5, 3, 12]
    """

    def __init__(self, decoder_layer, num_decoder):
        super(MyDecoder, self).__init__()
        self.decoders = clones_p(decoder_layer, num_decoder)
        self.norm = LayerNormP(decoder_layer.size)

    def forward(self, o_prev_decoder, o_encoder, src_mask, trg_mask):
        for decoder in self.decoders:
            o_prev_decoder = decoder(o_prev_decoder, o_encoder, src_mask, trg_mask)
        return o_prev_decoder


if __name__ == '__main__':
    # ##### params
    max_words_in_sentence = 4  # of words in each sentence
    batch_size = 5  # of sentences
    size_dict = 7  # size of word dictionary
    d_model = 12
    hidden_size_pff = 11
    num_head = 2
    dropout_rate = 0.1
    num_encoder_layer = 2

    # ##### data that is composed of sentence which is sequence of word index.
    np.random.seed(0)
    # gen_batch = data_gen(V=size_dict, batch=batch_size, nbatches=2, max_words_in_sentence=max_words_in_sentence)
    # batch0 = next(gen_batch)
    # batch0.src; batch0.src.shape
    # batch0.src_mask; batch0.src_mask.shape
    # batch0.trg
    # batch0.trg_mask
    src = torch.tensor([[0, 3, 0, 2],
                        [1, 0, 3, 2],
                        [0, 0, 0, 1],
                        [1, 0, 0, 1],
                        [3, 2, 2, 1]]); src
    src_mask = torch.tensor([[[1, 1, 1, 1]],
                             [[1, 1, 1, 1]],
                             [[1, 1, 1, 1]],
                             [[1, 1, 1, 1]],
                             [[1, 1, 1, 1]]]); src_mask
    # making input to encoder
    em = EmbeddingsP(d_model=d_model, vocab=size_dict)
    pe = PositionalEncodingP(d_model=d_model, dropout=0.)
    o_pe = pe(em(src))  # input to encoder, [5, 4, 12]
