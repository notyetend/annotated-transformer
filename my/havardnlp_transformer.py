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


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
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


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
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


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
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


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Fills elements of tensor with value where mask is one.
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    q = torch. @@@
    mha = MultiHeadedAttention(h=8, d_model=512)
    mha.forward()
    """
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # d_k = d_v = d_model/h
        self.h = h  # number of heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        """
        : 4 copies for W^Q, W^K, W^V and W^O (refers to page 5 of paper)
        : actually W^{Q, K, V} is for h heads, 
            so it's dim is d_model, not d_model/h
        """

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)
        """ex) when word dict length = 7 and embedding size = 6
        indexing 2nd word. 
        >>> nn.Embedding(7, 6)(torch.tensor(2))
        """
        self.d_model = d_model

    def forward(self, x):
        """Why multiply sqrt of d_model? maybe this is scaling factor
        that is sqrt of d_k in the eq1 of original paper.
        Btw d_k = d_v = d_model / h = 64,
        scaling factor is sqrt of length of vector itself.
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
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


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
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
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        encoder=Encoder(  # 'encoder' is stack of N 'encoder layer'
            layer=EncoderLayer(  # 'encoder layer' has 2 sub-layers,
                size=d_model,    # which are 'multi-attn' and 'position-wise fully connected feed-forward'
                self_attn=c(attn),
                feed_forward=c(ff),
                dropout=dropout
            ),
            N=N
        ),
        decoder=Decoder(  # stack of N 'decoder layer'
            layer=DecoderLayer(
                size=d_model,
                self_attn=c(attn),
                src_attn=c(attn),
                feed_forward=c(ff),
                dropout=dropout
            ),
            N=N
        ),
        src_embed=nn.Sequential(
            Embeddings(
                d_model=d_model, vocab=src_vocab
            ),
            c(position)  # positional encoding
        ),
        tgt_embed=nn.Sequential(  # Sequential(f1(x), f2) means y = f2(f1(x))
            Embeddings(
                d_model=d_model,
                vocab=tgt_vocab
            ),
            c(position)  # positional encoding
        ),
        generator=Generator(d_model=d_model, vocab=tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)  # fixed by kdw
    return model


class Batch:
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
    def make_std_mask(tgt, pad):
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
        tgt_mask = (tgt != pad).unsqueeze(-2)  # for example, turn [5, 3] to [5, 1, 3]
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
            subsequent_mask(size=tgt.size(-1)).type_as(tgt_mask.data)
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


def run_epoch(data_iter, model, loss_compute):
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


def batch_size_fn(new, count, sofar):
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


class NoamOpt:
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
    >>> opts = [NoamOpt(512, 1, 4000, None),
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


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
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


crit = LabelSmoothing(5, 0, 0.1)


def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    # print(predict)
    return crit(Variable(predict.log()),
                Variable(torch.LongTensor([1]))).data[0]

