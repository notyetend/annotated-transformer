from transformer.harvardnlp_transformer import *
from transformer.harvardnlp_transformer_keras import *
from transformer.test_config import *


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
    out_pe_p = pf(out_pe_p)

    # keras
    pe_k = PositionalEncodingK(d_model=d_model, dropout_rate=0.)
    out_pe_k = pe_k(K.constant(dummy_emb_np))
    out_pe_k = K.eval(out_pe_k)
    out_pe_k = pf(out_pe_k)

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
    out_ln_p = pf(out_ln_p.detach().numpy()[:])
    out_ln_p

    # keras
    ln_k = LayerNormK(features=d_model)
    out_ln_k = ln_k(K.constant(dummy_weight))
    out_ln_k = K.eval(out_ln_k)
    out_ln_k = pf(out_ln_k)
    out_ln_k

    # mean comparison
    mean_p_1 = torch.from_numpy(dummy_weight).mean(-1, keepdim=True).numpy()[:]
    mean_k_1 = K.mean(K.constant(dummy_weight), axis=-1, keepdims=True)
    assert np.array_equal(pf(mean_p_1), pf(K.eval(mean_k_1)))

    # std comparison
    """original implementation of havardnlp use unbiased std(default option for torch.tensor.std)
    as keras don't provide unbiased std, I changed torch code to use biased std."""
    std_p_1 = torch.from_numpy(dummy_weight).std(-1, keepdim=True).numpy()[:][0, :, :]  # unbiased
    std_k_1 = K.eval(K.std(K.constant(dummy_weight), axis=-1, keepdims=True))[0, :, :]  # biased
    assert not np.array_equal(pf(std_p_1), pf(std_k_1))

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
    assert np.array_equal(pf(std_p_2), pf(std_k_1))

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
    dummy_bias = np.zeros(num_out_node, ).astype('float32')
    dummy_weight = np.random.rand(num_out_node, num_in_node).astype('float32')
    dummy_input = np.random.rand(batch_size, max_words_in_sentence, num_in_node).astype('float32')
    dummy_output = np.matmul(dummy_input, np.transpose(dummy_weight)) + dummy_bias
    dummy_output

    # pytorch
    linear_p = nn.Linear(in_features=d_model, out_features=d_model)
    # pytorch takes weight matrix with shape of [num_out_node, num_in_node]
    linear_p.weight.data = torch.from_numpy(dummy_weight)
    linear_p.bias.data = torch.from_numpy(dummy_bias)

    assert np.array_equal(dummy_weight, linear_p.weight.detach().numpy())
    assert np.array_equal(dummy_bias, linear_p.bias.detach().numpy())
    out_linear_p = linear_p(torch.from_numpy(dummy_input)).detach().numpy()
    assert np.array_equal(pf(dummy_output), pf(out_linear_p))

    # keras
    input_k = Input(shape=(num_in_node,), batch_size=batch_size)
    linear_k = Dense(input_shape=(num_in_node,), units=num_out_node)
    linear_k(input_k)  # keras layer initialize weights after taking input
    linear_k.set_weights([dummy_weight.transpose(), dummy_bias])
    # keras takes weight matrix with shape of [num_in_node, num_out_node], so doing transpose
    assert np.array_equal(dummy_weight.transpose(), linear_k.get_weights()[0])  # get_weights() gives [weight, bias]
    assert np.array_equal(dummy_bias, linear_k.get_weights()[1])
    out_linear_k = K.eval(linear_k(K.constant(dummy_input)))
    assert np.array_equal(pf(dummy_output[0, 0]), pf(out_linear_k[0, 0]))

    dummy_output[0, 0]
    out_linear_k[0, 0]
    out_linear_p[0, 0]
    np.equal(out_linear_k[0, 0], out_linear_p[0, 0])
    assert np.array_equal(pf(out_linear_k, precision=3), pf(out_linear_p, precision=3))
    # no idea why they are differ in bigger precision ->
    """
    fyi
    multiplication of matrix with floats could be differ by order of operation.
    https://github.com/pytorch/pytorch/issues/17678
    """


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


def test_masked_fill():
    assert d_model % num_head == 0

    # dummy data
    d_k = d_v = d_model // num_head  # dim of query and key is d_k
    dummy_q_w_q = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k).astype('float32')
    print(dummy_q_w_q.shape)
    dummy_k_w_k = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k).astype('float32')
    dummy_v_w_v = np.random.rand(batch_size, num_head, max_words_in_sentence, d_v).astype('float32')
    dummy_batch = np.random.randint(low=1, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
    dummy_batch = dummy_batch.astype('int64')
    dummy_batch[:, 0] = 1
    pad = 0

    # pytorch - scores
    q_w_q_p = torch.from_numpy(dummy_q_w_q)
    k_w_k_p = torch.from_numpy(dummy_k_w_k)
    v_w_v_p = torch.from_numpy(dummy_v_w_v)
    d_k = q_w_q_p.size(-1)
    scores_p = torch.matmul(q_w_q_p, k_w_k_p.transpose(-2, -1)) / math.sqrt(d_k)
    scores_p
    scores_p.shape  # (5, 2, 4, 4), each dim represents (batch, head, tokes is sentence(query), tokens in sentence(key))

    # scores_p = torch.from_numpy(np.random.randint(low=1, high=max_words_in_sentence, size=(5, 2, 4, 4)).astype('float32'))

    # pytorch - masking
    src = torch.from_numpy(dummy_batch)
    trg = torch.from_numpy(dummy_batch)
    ba_p = BatchP(src=src, trg=trg, pad=pad)
    src_mask = ba_p.src_mask
    src_mask
    src_mask.shape  # (5, 1, 4)
    src_mask = src_mask.unsqueeze(1)
    src_mask.shape  # (5, 1, 1, 4)
    # (5, 1, 1, 4), each dim represents (batch, head, tokes is sentence(query), tokens in sentence(key))
    # dim(2) will be expanded to 4. this means we masking same keys for all query.
    scores_masked_p = scores_p.masked_fill(src_mask == 0, -1e9)
    # torch.mul(scores, (src_mask != 0).float()) + (src_mask == 0).float() * (-1e9)

    # pytorch - getting attention
    attn_p = F.softmax(scores_masked_p, dim=-1)
    torch.matmul(attn_p, v_w_v_p)[0, 0, 0, :]

    # keras - scores
    q_w_q_k = K.constant(dummy_q_w_q)
    k_w_k_k = K.constant(dummy_k_w_k)
    v_w_v_k = K.constant(dummy_v_w_v)
    d_k = q_w_q_k.shape.as_list()[-1]
    scores_k = K.batch_dot(q_w_q_k, k_w_k_k, axes=[3, 3]) / math.sqrt(d_k)
    assert np.array_equal(pf((scores_p.numpy())), pf(K.eval(scores_k)))

    # keras - masking
    src = K.constant(dummy_batch)
    trg = K.constant(dummy_batch)
    ba_k = BatchK(src=src, trg=trg, pad=pad)
    src_mask = ba_k.src_mask
    src_mask.shape
    src_mask = K.expand_dims(src_mask, axis=1)
    src_mask.shape

    def keras_masked_fill(x, mask, target_mask_val, filled_value=-1e9):
        return x * (x != target_mask_val) + (mask == target_mask_val) * filled_value

    scores_masked_k = keras_masked_fill(scores_k, src_mask, 0, -1e9)
    assert np.array_equal(pf(scores_masked_p.numpy()), pf(K.eval(scores_masked_k)))

    # keras - getting attention
    attn_k = K.softmax(scores_masked_k)
    assert np.array_equal(
        pf(F.softmax(scores_masked_p, dim=-1).numpy()),
        pf(K.eval(K.softmax(scores_masked_k)))
    )

    attn_k.shape
    K.eval(K.batch_dot(attn_k, v_w_v_k, axes=[3, 2]))[0, 0, 0, :]

    assert np.array_equal(
        pf(torch.matmul(attn_p, v_w_v_p).numpy(), precision=4),
        pf(K.eval(K.batch_dot(attn_k, v_w_v_k, axes=[3, 2])), precision=4)
    )


def test_attention():
    """
    ?_w_? represent tensor (query, key, value) passed through linear layer

    (query, key, value) have shape of (batch_size, max_words_in_sentence, d_model)
    ?_w_? have shape of (batch_size, num_head, max_words_in_sentence, d_model/num_head)
    """
    assert d_model % num_head == 0

    # dummy data
    d_k = d_v = d_model // num_head  # dim of query and key is d_k
    dummy_q_w_q = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k).astype('float32')
    print(dummy_q_w_q.shape)
    dummy_k_w_k = np.random.rand(batch_size, num_head, max_words_in_sentence, d_k).astype('float32')
    dummy_v_w_v = np.random.rand(batch_size, num_head, max_words_in_sentence, d_v).astype('float32')
    dummy_batch = np.random.randint(low=1, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
    dummy_batch = dummy_batch.astype('int64')
    dummy_batch[:, 0] = 1
    pad = 0

    # pytorch
    q_w_q_p = torch.from_numpy(dummy_q_w_q)
    k_w_k_p = torch.from_numpy(dummy_k_w_k)
    v_w_v_p = torch.from_numpy(dummy_v_w_v)
    src = torch.from_numpy(dummy_batch)
    trg = torch.from_numpy(dummy_batch)
    ba_p = BatchP(src=src, trg=trg, pad=pad)
    src_mask_p = ba_p.src_mask
    src_mask_p = src_mask_p.unsqueeze(1)
    x_p, attn_p = attention_p(q_w_q_p, k_w_k_p, v_w_v_p, src_mask_p)

    # keras
    q_w_q_k = K.constant(dummy_q_w_q)
    k_w_k_k = K.constant(dummy_k_w_k)
    v_w_v_k = K.constant(dummy_v_w_v)
    src = K.constant(dummy_batch)
    trg = K.constant(dummy_batch)
    ba_k = BatchK(src=src, trg=trg, pad=pad)
    src_mask_k = ba_k.src_mask
    src_mask_k = K.expand_dims(src_mask_k, axis=1)
    x_k, attn_k = attention_k(q_w_q_k, k_w_k_k, v_w_v_k, src_mask_k)

    assert np.array_equal(
        pf(x_p.numpy(), 4),
        pf(K.eval(x_k), 4)
    )


def test_reshape_vs_permutate():
    """

    view in pytorch == reshape in keras

    transpose in pytorch == permute_dimensions in keras
    (if number of dim is 2, transpose in keras works same as permute_dimensions in keras)

    view나 reshape은 대단히 조심히 써야한다.
    그냥 차원을 맞추기 위해 다음 row의 원소들을 그냥 가져와 붙이기 때문이다.

    reshape으로 차원간 순서를 바꾸는 짓을 하면 안된다.
    단지 차원을 추가하거나 차원을 쪼게는 용도로만 써야한다.
    """
    dim0 = 3
    dim1 = 4
    a = np.random.randint(low=1, high=max_words_in_sentence, size=(dim0, dim1))
    a

    # pytorch
    ap = torch.from_numpy(a)
    ap1 = ap.view(dim1, dim0)
    ap2 = ap1.transpose(0, 1)
    assert np.array_equal(ap1.transpose(0, 1), ap1.transpose(1, 0))

    # keras
    ak = K.constant(a)
    ak1 = K.reshape(ak, (dim1, dim0))
    assert np.array_equal(ap1.numpy(), K.eval(ak1))
    ak2 = K.permute_dimensions(ak1, pattern=(1, 0))
    assert np.array_equal(ap2.numpy(), K.eval(ak2))


def test_multi_head_attention():
    assert d_model % num_head == 0

    # dummy data
    np.random.seed(0)
    dummy_q = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')
    dummy_k = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')
    dummy_v = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')
    dummy_mask = np.ones((batch_size, 1, max_words_in_sentence)).astype('int64')
    dummy_biases = [np.zeros(d_model, ).astype('float32') for _ in range(4)]
    dummy_weights = [np.random.rand(d_model, d_model).astype('float32') for _ in range(4)]

    # pytorch
    q_p = torch.from_numpy(dummy_q)
    k_p = torch.from_numpy(dummy_k)
    v_p = torch.from_numpy(dummy_v)
    src_mask_p = torch.from_numpy(dummy_mask)

    q_linear_p = nn.Linear(d_model, d_model)
    q_linear_p.bias.data = torch.from_numpy(dummy_biases[0])
    q_linear_p.weight.data = torch.from_numpy(dummy_weights[0])
    q_w_q_p = q_linear_p(q_p)  # (5, 4, 12)
    q_w_q_p = q_w_q_p.view(batch_size, -1, num_head, d_model // num_head)  # (5, 4, 2, 6)
    q_w_q_p = q_w_q_p.transpose(1, 2)  # (5, 2, 4, 6)

    k_linear_p = nn.Linear(d_model, d_model)
    k_linear_p.bias.data = torch.from_numpy(dummy_biases[1])
    k_linear_p.weight.data = torch.from_numpy(dummy_weights[1])
    k_w_k_p = k_linear_p(k_p)  # (5, 4, 12)
    k_w_k_p = k_w_k_p.view(batch_size, -1, num_head, d_model // num_head)  # (5, 4, 2, 6)
    k_w_k_p = k_w_k_p.transpose(1, 2)  # (5, 2, 4, 6)

    v_linear_p = nn.Linear(d_model, d_model)
    v_linear_p.bias.data = torch.from_numpy(dummy_biases[2])
    v_linear_p.weight.data = torch.from_numpy(dummy_weights[2])
    v_w_v_p = v_linear_p(v_p)  # (5, 4, 12)
    v_w_v_p = v_w_v_p.view(batch_size, -1, num_head, d_model // num_head)  # (5, 4, 2, 6)
    v_w_v_p = v_w_v_p.transpose(1, 2)  # (5, 2, 4, 6)

    x_p, attn_p = attention_p(q_w_q_p, k_w_k_p, v_w_v_p, src_mask_p.unsqueeze(1))
    x_p.shape  # (5, 2, 4, 6)
    attn_p.shape  # (5, 2, 4, 4)
    # contiguous function makes tensor array stored contiguously in memory
    # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    x_return_p = x_p.transpose(1, 2).contiguous().view(batch_size, -1, d_model)

    # keras
    q_k = K.constant(dummy_q)
    k_k = K.constant(dummy_k)
    v_k = K.constant(dummy_v)
    src_mask_k = K.constant(dummy_mask)

    q_linear_k = Dense(input_shape=(d_model,), units=d_model)
    q_w_q_k = q_linear_k(q_k)  # (5, 4, 12)
    # not like pytorch, it's only possible to set weight after input passed through layer.
    q_linear_k.set_weights([dummy_weights[0].transpose(), dummy_biases[0]])
    q_w_q_k1 = K.reshape(q_w_q_k, (batch_size, -1, num_head, d_model // num_head))  # (5, 4, 2, 6)
    q_w_q_k = K.permute_dimensions(q_w_q_k1, pattern=(0, 2, 1, 3))  # (5, 2, 4, 6)
    assert K.eval(K.reshape(q_w_q_k1, (batch_size, num_head, -1, d_model // num_head))).shape == \
           K.eval(K.permute_dimensions(q_w_q_k1, pattern=(0, 2, 1, 3))).shape
    assert not np.array_equal(
        K.reshape(q_w_q_k, (batch_size, num_head, -1, d_model // num_head)),
        K.eval(K.permute_dimensions(q_w_q_k, pattern=(0, 2, 1, 3)))
    )  # result of reshape and permute_dimension have same shape but not about their values.
    # and reshape do not give what you actually wanted.

    k_linear_k = Dense(input_shape=(d_model,), units=d_model)
    k_w_k_k = k_linear_k(k_k)  # (5, 4, 12)
    k_linear_k.set_weights([dummy_weights[1].transpose(), dummy_biases[1]])
    k_w_k_k = K.reshape(k_w_k_k, (batch_size, -1, num_head, d_model // num_head))  # (5, 4, 2, 6)
    k_w_k_k = K.permute_dimensions(k_w_k_k, pattern=(0, 2, 1, 3))  # (5, 2, 4, 6)

    v_linear_k = Dense(input_shape=(d_model,), units=d_model)
    v_w_v_k = v_linear_k(v_k)  # (5, 4, 12)
    v_linear_k.set_weights([dummy_weights[2].transpose(), dummy_biases[2]])
    v_w_v_k = K.reshape(v_w_v_k, (batch_size, -1, num_head, d_model // num_head))  # (5, 4, 2, 6)
    v_w_v_k = K.permute_dimensions(v_w_v_k, pattern=(0, 2, 1, 3))

    x_k, attn_k = attention_k(q_w_q_k, k_w_k_k, v_w_v_k, K.expand_dims(src_mask_k, axis=1))
    x_k.shape  # (5, 2, 4, 6)
    attn_k.shape  # (5, 2, 4, 4)
    x_return_k = K.reshape(K.permute_dimensions(x_k, pattern=(0, 2, 1, 3)), shape=(batch_size, -1, d_model))

    # comparing output of linear and shape exchange
    assert np.array_equal(pf(x_p.detach().numpy(), 3), pf(K.eval(x_k), 3))
    assert np.array_equal(pf(x_return_p.detach().numpy(), 3), pf(K.eval(x_return_k), 3))

    # pytorch
    o_linear_p = nn.Linear(d_model, d_model)
    o_linear_p.bias.data = torch.from_numpy(dummy_biases[3])
    o_linear_p.weight.data = torch.from_numpy(dummy_weights[3])

    linears_p = [q_linear_p, k_linear_p, v_linear_p, o_linear_p]
    mha_p = MultiHeadedAttentionP(h=2, d_model=d_model, dropout=0., linears=linears_p)
    o_mha_p = mha_p(q_p, k_p, v_p, src_mask_p)

    # keras
    o_linear_k = Dense(input_shape=(d_model,), units=d_model)
    o_linear_k(q_k)  # dummy input to set weights
    o_linear_k.set_weights([dummy_weights[3].transpose(), dummy_biases[3]])
    linears_k = [q_linear_k, k_linear_k, v_linear_k, o_linear_k]
    mha_k = MultiHeadedAttentionK(h=2, d_model=d_model, dropout=0., linears=linears_k)
    o_mha_k = mha_k([q_k, k_k, v_k, src_mask_k])

    assert np.array_equal(pf(o_mha_p.detach().numpy()), pf(K.eval(o_mha_k)))


def test_position_wise_feed_forward():
    dummy_emb_np = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')

    # pytorch
    input_p = torch.from_numpy(dummy_emb_np)
    pff_p = PositionwiseFeedForwardP(d_model, hidden_size_pff)
    pff_p(input_p)

    # keras
    input_k = K.constant(dummy_emb_np)
    pff_k = PositionwiseFeedForwardK(d_model, hidden_size_pff)
    pff_k(input_k)


def test_sublayer_connection():
    dummy_mha_out = np.random.rand(batch_size, max_words_in_sentence, d_model).astype('float32')

    # pytorch
    input_p = torch.from_numpy(dummy_mha_out)
    slc_p = SublayerConnectionP(size=d_model, dropout=0.)
    mha_p = MultiHeadedAttentionP(h=num_head, d_model=d_model, dropout=0.)
    output_slc_p = slc_p(x=input_p, sublayer=lambda x: mha_p(x, x, x))

    # keras
    input_k = K.constant(dummy_mha_out)
    mha_k = MultiHeadedAttentionK(h=num_head, d_model=d_model, dropout=0.)
    slc_k = SublayerConnectionK(size=d_model, sublayer=lambda x: mha_k(x, x, x), dropout=0.)
    output_slc_k = slc_k(input_k)
    # ToDo: Need to fix error that is '__call__() takes 2 positional arguments but 4 were given'


def test_make_model(src_vocab, trg_vocab, num_coder_blocks, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """

    Parameters
    ----------
    src_vocab: size of source vocab dict
    trg_vocab: size of target vocab dict
    num_coder_blocks: number of encoder/decoder block
    d_model: embedding size
    d_ff: number of hidden layer node
    h: number of head
    dropout: dropout rate

    Returns
    -------
    model
    """

    class TestEncoderDecoderP(nn.Module):
        def __init__(self):
            super(TestEncoderDecoderP, self).__init__()

        def forward(self, src, trg, src_mask, trg_mask):
            # input to encoder (embedding and positional encoding)
            src_emb_layer_p = EmbeddingsP(d_model=d_model, vocab=src_vocab)
            src_pe_p = PositionalEncodingP(d_model=d_model, dropout=dropout)
            input_encoder = src_pe_p(src_emb_layer_p(src))  # (5, 4, 12)

            # encoder
            for _ in range(num_coder_blocks):
                # multi headed attention and 1st sublayer connection
                mha_p = MultiHeadedAttentionP(h=h, d_model=d_model, dropout=dropout)
                self_attn = lambda x: mha_p(x, x, x, src_mask)
                slc1_p = SublayerConnectionP(size=d_model, dropout=dropout)
                out_slc1_p = slc1_p(x=input_encoder, sublayer=self_attn)

                # position wise feed forward and 2nd sublayer connection
                pff_p = PositionwiseFeedForwardP(d_model, d_ff)
                slc2_p = SublayerConnectionP(size=d_model, dropout=dropout)
                input_encoder = slc2_p(x=out_slc1_p, sublayer=pff_p)
            output_encoder = LayerNormP(features=d_model)(input_encoder)

            # input to decoder (embedding and positional encoding)
            trg_emb_layer_p = EmbeddingsP(d_model=d_model, vocab=trg_vocab)
            trg_pe_p = PositionalEncodingP(d_model=d_model, dropout=dropout)
            input_decoder = trg_pe_p(trg_emb_layer_p(trg))  # (5, 4, 12)

            # decoder
            for _ in range(num_coder_blocks):
                # sublayer 1 of decoder
                mha1_p = MultiHeadedAttentionP(h=h, d_model=d_model, dropout=dropout)
                self_attn = lambda x: mha1_p(x, x, x, trg_mask)
                slc1_p = SublayerConnectionP(size=d_model, dropout=dropout)
                out_slc1_p = slc1_p(x=input_decoder, sublayer=self_attn)

                # sublayer 2 of decoder
                mha2_p = MultiHeadedAttentionP(h=h, d_model=d_model, dropout=dropout)
                src_attn = lambda x: mha2_p(x, output_encoder, output_encoder, src_mask)
                slc2_p = SublayerConnectionP(size=d_model, dropout=dropout)
                out_slc2_p = slc2_p(x=out_slc1_p, sublayer=src_attn)

                # position wise feed forward and 2nd sublayer connection
                pff_p = PositionwiseFeedForwardP(d_model, d_ff)
                slc3_p = SublayerConnectionP(size=d_model, dropout=dropout)
                input_encoder = slc3_p(x=out_slc2_p, sublayer=pff_p)
            output_decoder = LayerNormP(features=d_model)(input_encoder)

            return output_decoder

    model = TestEncoderDecoderP()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def test_get_model_output(batch=None):
    """
    Parameters
    ----------
    batch: element of batch instance. next(Batch(...))
            this is used as input of encoder
    Returns
    -------
    output of decoder
    """
    pad = 0

    # pytorch ###########################
    # test data
    if batch:
        ba_p = batch
    else:
        dummy_batch = np.random.randint(low=1, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence))
        dummy_batch[:, 0] = 1
        dummy_batch = dummy_batch.astype('int64')

        src = torch.from_numpy(dummy_batch)
        trg = torch.from_numpy(dummy_batch)
        ba_p = BatchP(src=src, trg=trg, pad=pad)

    model = test_make_model(size_dict, size_dict, 2, d_model, hidden_size_pff, num_head, 0.)
    out = model.forward(ba_p.src, ba_p.trg, ba_p.src_mask, ba_p.trg_mask)
    # This is output of decoder


def test_Label_smoothing():
    """
    this is used in SimpleLossCompute
    """
    dummy_out_decoder = np.random.rand(batch_size, max_words_in_sentence - 1, d_model).astype('float32')
    dummy_target_y = np.random.randint(low=0, high=max_words_in_sentence, size=(batch_size, max_words_in_sentence - 1))

    out = torch.from_numpy(dummy_out_decoder)
    trg_y = torch.from_numpy(dummy_target_y)

    generator = GeneratorP(d_model=d_model, vocab=size_dict)

    x = generator(out)  # (5, 3, 7), which is (batch_size, max_words_in_sentence - 1, size_dict)
    x = x.contiguous().view(-1, x.size(-1))  # (15, 7)
    target = trg_y.contiguous().view(-1)  # (15)

    norm = target.size(-1)  # 15
    size = size_dict
    padding_idx = 0
    smoothing = 0.2

    # from here, forward function of LabelSmoothingP
    true_dist = x.data.clone()  # (15, 7)
    true_dist.fill_(smoothing / (size - 2))
    """
        15개 token 위치에 대한 softmax 값을 담고 있는 (15, 7) 행렬을 
        전부 (smoothing / size-2))로 체운 후
    """

    true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)
    """
        15개 token에 대한 정답지인 target에 해당하는 index에 (1 - smoothing)을 체운다.
        즉 오답의 하한(smoothing / (size-2))과 정답의 상한값(1-smoothing)을 체운 것이다.
    """

    true_dist[:, padding_idx] = 0
    """
        padding_idx에 해당하는 컬럼은 그 값이 모두 0이고
        이 컬럼에 대해서는 smoothing 하지 않을 것이다.
        이를 위해 true_dist에도 이 컬럼(padding column)에는 0을 체운다.
    """

    mask = torch.nonzero(target.data == padding_idx)  # getting idx of nonzero
    """
    target token이 padding token에 해당하는 index를 찾는다.
    """

    if mask.dim() > 1:  # target token이 padding token에 해당하는 값이 있으면
        true_dist.index_fill_(0, mask.squeeze(), 0.0)
        """
            target token이 0(padding_idx)인 경우
            해당하는 row에 대해서는 smoothing 하지 않을 것이므로 true_dist를 모두 0으로 설정
        """

    loss = nn.KLDivLoss(reduction='sum')(x, Variable(true_dist, requires_grad=False))
    loss = loss / norm
    loss.backward()


def test_greedy_decode():
    class SimpleLossCompute:
        "A simple loss compute and train function."

        def __init__(self, generator, criterion, opt=None):
            self.generator = generator
            self.criterion = criterion
            self.opt = opt

        def __call__(self, x, y, norm):
            x = self.generator(x)
            norm = norm.type(torch.FloatTensor)  # added by kdw
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                                  y.contiguous().view(-1)) / norm
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
            return loss.data[0] * norm

    V = 11
    criterion = LabelSmoothingP(size=V, padding_idx=0, smoothing=0.0)
    model = make_model_p(V, V, N=2)
    model_opt = NoamOptP(model.src_embed[0].d_model, 1, 400,
                         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch_p(data_gen_p(V, 30, 20, 10), model,
                    SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch_p(data_gen_p(V, 30, 5, 10), model,
                          SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode_p(model, src, src_mask, max_len=10, start_symbol=1))


test_greedy_decode()
