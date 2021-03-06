# Standard PyTorch imports
import time
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt


class EncoderDecoderP(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoderP, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory,
                              src_mask, tgt_mask)
        return output

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class GeneratorP(nn.Module):
    """
    Define standard linear + softmax generation step.

    ???
    """

    def __init__(self, d_model, vocab):
        super(GeneratorP, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones_p(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderP(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(EncoderP, self).__init__()
        self.layers = clones_p(layer, N)
        self.norm = LayerNormP(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNormP(nn.Module):
    """
    Construct a layernorm module (See citation for details).

    ln = LayerNorm(10)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).type(torch.float)
    ln.forward(x)


    """

    def __init__(self, features, eps=1e-6):
        super(LayerNormP, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)  # changed @@@
        """
        default value of unbiased is True. because Keras only provide biased std. 
        for comparison purpose, changed to use biased std.
        """
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnectionP(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnectionP, self).__init__()
        self.norm = LayerNormP(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.

        x : input to sublayer
        sublayer: {multi-head attention or position-wise feed-forward}
        Following original paper,
        this should be norm(x + dropout(sublayer(x)))


        But according to below comment
        https://github.com/tensorflow/tensor2tensor/blob/v1.6.5/tensor2tensor/layers/common_hparams.py#L110-L112
        # TODO(noam): The current settings ("", "dan") are the published version
        # of the transformer.  ("n", "da") seems better for harder-to-learn
        # models, so it should probably be the default.
        and comment below
         https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
         The sequence is specified as a string which may contain the following
          characters:
            a: add previous_value
            n: apply normalization
            d: apply dropout
            z: zero add
        original paper use only post process 'dan'
            which is doing dropout, add, norm for sublayer_output
            and it is norm(x + dropout(sublayer(x))

        enhanced version is doing norm on x and then dropout on sublayer_output and add.
            this is x + dropout(sublayer(norm(x)))
        """

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayerP(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)


    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayerP, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones_p(SublayerConnectionP(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderP(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(DecoderP, self).__init__()
        self.layers = clones_p(layer, N)
        self.norm = LayerNormP(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayerP(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayerP, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones_p(SublayerConnectionP(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask_p(size):
    """
    Mask out subsequent positions.

    >>> subsequent_mask(3)
    tensor([
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]
            ]], dtype=torch.uint8)  # [1, 3, 3]

    This function gives mask for a sentence with 'size' words.

    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention_p(q_w_q, k_w_k, v_w_v, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'

    With settings
        d_model = 12
        num of word in sentence = 4
        num of sentences = nbatches = 5
        num of heads = self.h = 2
        self.d_k = d_model / self.h = 6

    shape of q_w_q, k_w_k, v_w_v: [5, 2, 4, 6]
        -> last dim is 6 because we divided d_model by num heads. (12/6)
        -> [num sentences, heads, num words, word info 'for this head']

    torch.matmul(q_w_q, k_w_k.transpose(-2, -1)) / math.sqrt(d_k)
        -> attention(QW^Q, KW^K, VW^V) = softmax( {QW^Q dot (KW^K)^T} / sqrt(d_k)} ) dot VW^V
        -> eq (1) of original paper

    shape of scores = torch.matmul(q_w_q, k_w_k.transpose(-2, -1)): [5, 2, 4, 4]
        -> dot product of [5, 2, 4, 6] and [5, 2, 6, 4]

    shape of mask: [5, 1, 1, 4]
        -> this mask shape will be extended as shape of scores, scores's shape is [5, 2, 4, 4]

    scores.masked_fill(mask == 0, -1e9)
        -> create new tensor with scores having original value if mask != 0 otherwise having -1e9 if mask == 0.
    """
    d_k = q_w_q.size(-1)
    scores = torch.matmul(q_w_q, k_w_k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Fills scores where mask is zero
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v_w_v), p_attn


class MultiHeadedAttentionP(nn.Module):
    """

    """
    def __init__(self, h, d_model, dropout=0.1, linears=None):
        """Take in model size and number of heads.

        h: number of heads
        linears: added by kdw to test fixed weight linear.
        """
        super(MultiHeadedAttentionP, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # d_k = d_v = d_model/h
        self.h = h  # number of heads

        if linears:
            assert len(linears) == 4
            self.linears = linears
        else:
            self.linears = clones_p(nn.Linear(d_model, d_model), 4)

        """
        : 4 copies for W^Q, W^K, W^V and W^O (refers to page 5 of paper)
        : actually W^{Q, K, V} is for h heads, 
            so it's dim is d_model, not d_model/h
        """

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2

        With settings
            d_model = 12
            num of word in sentence = 4
            num of sentences = nbatches = 5
            num of heads = self.h = 2
            self.d_k = d_model / self.h = 6

        shape of query: [5, 4, 12]
        shape of key:   [5, 4, 12]
        shape of value: [5, 4, 12]

        shape of w_q, w_k, w_v: [12, 12]

        shape of    self.linears[0](query): [5, 4, 12]
                    self.linears[1](key)  : [5, 4, 12]
                    self.linears[2](value): [5, 4, 12]

        ※ view function is similar to reshape of numpy

        shape of self.linears[0](query).view(5, -1, 2, 6): [5, 4, 2, 6]
         -> splitting last dim into 2

        shape of q_w_q = self.linears[0](query).view(5, -1, 2, 6).transpose(1, 2): [5, 2, 4, 6]
                 k_w_k = self.linears[1](key  ).view(5, -1, 2, 6).transpose(1, 2): [5, 2, 4, 6]
                 v_w_v = self.linears[2](value).view(5, -1, 2, 6).transpose(1, 2): [5, 2, 4, 6]
         -> exchanging 1st and 2nd dim to ... ??? why???
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q_w_q, k_w_k, v_w_v = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention_p(q_w_q, k_w_k, v_w_v, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        """
        shape of x: [5, 2, 4, 6]  -> last two dim represent one sentence.
                 x.transpose(1, 2): [5, 4, 2, 6] -> last two dim represent attentions(multiple heads) of one word.
                 x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k): [5, 4, 12]
                    -> concatenating attentions into one.
                    
        self.linears[-1](x) -> concat(head1, head2, ...) dot W^O
            shape of W^O: d_model times d_K
        """
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForwardP(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EmbeddingsP(nn.Module):
    """
    >>> np.random.seed(0)
    >>> emb_weight = np.random.rand(7, 12)  # total 7 tokens and hidden size is 12
    >>> em = EmbeddingsP(d_model=12, vocab=7, weight=emb_weight)
    >>> test_emb_pytorch = em(torch.tensor([list(range(7))]))
    >>> test_emb_pytorch = test_emb_pytorch.numpy()[:]
    >>> print(test_emb_pytorch, test_emb_pytorch.shape)
    """
    def __init__(self, d_model, vocab, weight=None):
        super(EmbeddingsP, self).__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)

        if weight is None:
            pass
        elif isinstance(weight, np.ndarray):
            self.lut.weight.data.copy_(torch.from_numpy(weight))
            self.lut.weight.requires_grad = False
        else:
            raise ValueError('Invalid weight')

    def forward(self, x):
        """Why multiply sqrt of d_model? maybe this is scaling factor
        that is sqrt of d_k in the eq1 of original paper.
        Btw d_k = d_v = d_model / h = 64,
        scaling factor is sqrt of length of vector itself.
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncodingP(nn.Module):
    """Implement the PE function.

    >>> # test implementation
    >>> max_len = 4
    >>> d_model = 12
    >>> pe = torch.zeros(max_len, d_model); print(pe, pe.shape)

    >>> # sentence number 0 to max_len-1
    >>> position = torch.arange(start=0.0, end=max_len).unsqueeze(1); print(position, position.shape)

    >>> # div term = exp{ -frac{2i}{d_model} * log(10000) }
    >>> # PE(pos, 2i) = sin(pos/10000^{frac{2i}{d_model}}
    >>> # i : index of 0 to d_model-1
    >>> div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    >>> pe[:, 0::2] = torch.sin(position * div_term)
    >>> pe[:, 1::2] = torch.cos(position * div_term)
    >>> pe.shape  # (4, 12)
    >>> pe = pe.unsqueeze(0)
    >>> pe.shape  # (1, 4, 12)


    >>> # plotting
    >>> d_model = 12
    >>> num_sentences = 1
    >>> num_tokens_in_sentence = 100
    >>> plt.figure(figsize=(15, 5))
    >>> pe = PositionalEncodingP(d_model=d_model, dropout=0.)
    >>> y = pe.forward(Variable(torch.zeros(1, num_tokens_in_sentence, d_model)))
    >>> plt.plot(np.arange(num_tokens_in_sentence), y[0, :, 4:8].data.numpy())
    >>> plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    >>> plt.show()
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncodingP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def make_model_p(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.

    :param src_vocab:
    :param tgt_vocab:
    :param N:           # of transformer block
    :param d_model:     size of embedding
    :param d_ff:        hidden layer size of the 'Position wize feed forward' unit.
    :param h:           # of heads
    :param dropout:
    :return:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttentionP(h, d_model)
    ff = PositionwiseFeedForwardP(d_model, d_ff, dropout)
    position = PositionalEncodingP(d_model, dropout)

    model = EncoderDecoderP(
        encoder=EncoderP(  # 'encoder' is stack of N 'encoder layer'
            layer=EncoderLayerP(  # 'encoder layer' has 2 sub-layers,
                size=d_model,    # which are 'multi-attn' and 'position-wise fully connected feed-forward'
                self_attn=c(attn),
                feed_forward=c(ff),
                dropout=dropout
            ),
            N=N
        ),
        decoder=DecoderP(  # stack of N 'decoder layer'
            layer=DecoderLayerP(
                size=d_model,
                self_attn=c(attn),
                src_attn=c(attn),
                feed_forward=c(ff),
                dropout=dropout
            ),
            N=N
        ),
        src_embed=nn.Sequential(
            EmbeddingsP(
                d_model=d_model, vocab=src_vocab
            ),
            c(position)  # positional encoding
        ),
        tgt_embed=nn.Sequential(  # Sequential(f1(x), f2) means y = f2(f1(x))
            EmbeddingsP(
                d_model=d_model,
                vocab=tgt_vocab
            ),
            c(position)  # positional encoding
        ),
        generator=GeneratorP(d_model=d_model, vocab=tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)  # fixed by kdw
    return model


class BatchP:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        """
        >>> trg = src = torch.tensor([[1, 2, 3, 2],
                                     [1, 2, 1, 4],
                                     [1, 2, 4, 5],
                                     [1, 1, 2, 1],
                                     [1, 2, 5, 5]])  # [5, 4]
        >>> pad = 0
        
        >>> src.unsqueeze(0).shape   # (1, 5, 10)
        >>> src.unsqueeze(1).shape   # (5, 1, 10)
        >>> src.unsqueeze(-1).shape  # (5, 10, 1)
        >>> src.unsqueeze(-2).shape  # (5, 1, 10)
        """
        if trg is not None:
            self.trg = trg[:, :-1]  # without last column, why???
            self.trg_y = trg[:, 1:]  # without first column, why???
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()  # where this used for???

    @staticmethod
    def make_std_mask(trg, pad):
        """
        Create a mask to hide padding and future words.

        >>> trg = src = torch.tensor([[1, 2, 3, 2],
                                      [1, 2, 1, 4],
                                      [1, 2, 4, 5],
                                      [1, 1, 2, 1],
                                      [1, 2, 5, 5]])  # (5, 4), number of sentence in this batch is 5, size of embedding is 4.

        >>> tgt = trg[:, :-1]
        >>> pad = 0
        """
        tgt_mask = (trg != pad).unsqueeze(-2)  # for example, turn [5, 3] to [5, 1, 3]
        """
        >>> (tgt != pad).shape  # (5, 3)
        >>> tgt_mask = (tgt != pad).unsqueeze(-2)  # [5, 1, 3]
        >>> tgt_mask
        tensor([[[1, 1, 1]],  # mask for 1st sentence
                [[1, 1, 1]],  #          2nd
                [[1, 1, 1]],  #          3rd
                [[1, 1, 1]],  #          4th
                [[1, 1, 1]]], #          5th
                dtype=torch.uint8)  # [5, 1, 3]    
        """
        tgt_mask = tgt_mask & Variable(
            subsequent_mask_p(size=trg.size(-1)).type_as(tgt_mask.data)
        )
        """
        >>> tgt.size(-1)  # 3
        
        >>> v = Variable(subsequent_mask(size=tgt.size(-1)).type_as(tgt_mask.data))
        >>> v  
        tensor([[[1, 0, 0],
                 [1, 1, 0],
                 [1, 1, 1]]], dtype=torch.uint8)  # [1, 3, 3]
                
        >>> tgt_mask & v
        tensor([[[1, 0, 0],  # mask for 1st sentence
                 [1, 1, 0],
                 [1, 1, 1]],
                [[1, 0, 0],  # mask for 2nd sentence
                 [1, 1, 0],
                 [1, 1, 1]],
                [[1, 0, 0],  # mask for 3rd sentence
                 [1, 1, 0],
                 [1, 1, 1]],
                [[1, 0, 0],  # mask for 4th sentence
                 [1, 1, 0],
                 [1, 1, 1]],
                [[1, 0, 0],  # mask for 5th sentence
                 [1, 1, 0],
                 [1, 1, 1]]], dtype=torch.uint8)  # [5, 3, 3]
        """
        return tgt_mask


def run_epoch_p(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f"
                  % (i, loss / batch.ntokens.type(torch.FloatTensor), tokens.type(torch.FloatTensor) / elapsed))
            start = time.time()
            tokens = 0
    # return total_loss / total_tokens
    return total_loss / total_tokens.type(torch.FloatTensor)  # fixed by kdw


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn_p(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOptP:
    """Optim wrapper that implements rate.

    from paper 5.3 Optimizer.
    We used the Adam optimizer with β1 = 0:9, β2 = 0:98 and epsilon = 10−9.
    We varied the learning rate over the course of training, according to the formula: (3)
    
    This corresponds to increasing the learning rate linearly 
     for the first warmup_steps training steps,
     and decreasing it thereafter proportionally 
     to the inverse square root of the step number. 
     We used warmup_steps = 4000

    ex)
    >>> opts = [NoamOptP(512, 1, 4000, None),
                NoamOpt(512, 1, 8000, None),
                NoamOpt(256, 1, 4000, None)]
    >>> plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    >>> plt.legend(["512:4000", "512:8000", "256:4000"])
    >>> plt.show()
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup  # before this steps, lr increase | after then lr decrease.
        self.factor = factor  # ???
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()  # learning rate actually we use.
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate  # save to be used for calculating lr in the next step.
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step

        # formula (3) of the original paper.
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt_p(model):
    return NoamOptP(model.src_embed[0].d_model, 2, 4000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothingP(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingP, self).__init__()
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.criterion = nn.KLDivLoss(reduction='sum')  # fixed by kdw
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 1:  # fixed 0 to 1 by kew.
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def greedy_decode_p(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask_p(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def data_gen_p(V, batch, nbatches, max_words_in_sentence):
    """
    Generate random data for a src-tgt copy task.

    # 5: # of sentences per batch == batch(2nd arg)
    # 4: # of words in each sentence
    # 7: size of word dictionary
    np.random.randint(low=1, high=7, size=(5, 4))  # 5 by 4 matrix

    >>> gen = data_gen(7, 5, 2, 4)
    >>> batch0 = next(gen)
    >>> batch0.src
    >>> batch0.trg
    >>> batch0.src.shape  # [5, 4]
    >>> batch0.ntokens  # 15
    tensor([[1, 2, 3, 2],
            [1, 2, 1, 4],
            [1, 2, 4, 5],
            [1, 1, 2, 1],
            [1, 2, 5, 5]])  # [5, 4]
    >>> batch0.src_mask
    tensor([[[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1]]], dtype=torch.uint8)  # [5, 1, 4]
    >>> batch0.trg
    tensor([[1, 2, 3],
            [1, 2, 1],
            [1, 2, 4],
            [1, 1, 2],
            [1, 2, 5]])  # [5, 3]
    >>> batch0.trg_y
    tensor([[2, 3, 2],
            [2, 1, 4],
            [2, 4, 5],
            [1, 2, 1],
            [2, 5, 5]])  # [5, 3]
    >>> batch0.trg_mask
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

    >>> batch0.ntokens  # 15

    >>> batch0.src.shape  # (5, 4)
    """
    for _ in range(nbatches):
        data = torch.from_numpy(np.random.randint(low=1, high=V, size=(batch, max_words_in_sentence)))
        data[:, 0] = 1  # 1 for first column
        src = Variable(data, requires_grad=False).type(torch.long)
        tgt = Variable(data, requires_grad=False).type(torch.long)
        yield BatchP(src, tgt, 0)
