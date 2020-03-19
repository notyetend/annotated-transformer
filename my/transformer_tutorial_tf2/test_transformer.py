from transformer_tutorial_tf2.transformer import (
    train_examples, val_examples,
    tokenizer_en, tokenizer_pt,
    get_dataset, encode, tf_encode, filter_max_length
)

from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter


def test_dataset_adapter():
    train_examples, val_examples  # tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter
    print(train_examples, val_examples)

    pt, en = next(train_examples.__iter__())
    print('pt in train example:', pt.numpy())
    print('en in train example:', en.numpy())


def test_tokenizer():
    tokenizer_pt, tokenizer_en  # tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder
    print('num tokens:', tokenizer_en.vocab_size)
    print('token list:', tokenizer_en.subwords)
    print('1st token:', tokenizer_en.decode([0]), tokenizer_en.subwords[0])

    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))
    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


def test_encode():
    pt, en = next(train_examples.__iter__())
    lang1, lang2 = encode(pt, en)
    print('encoded lang1:', lang1)
    print('encoded lang2:', lang2)

    print('decoded lang1:', tokenizer_pt.decode(lang1[1:-1]))
    print('decoded lang2:', tokenizer_en.decode(lang2[1:-1]))


def test_tf_encode():
    pt, en = next(train_examples.__iter__())
    tf_encode(pt, en)


def test_filter_max_length():
    pt, en = next(train_examples.__iter__())
    x, y = tf_encode(pt, en)
    filter_max_length(x, y)


def test_get_dataset():
    train_dataset, val_dataset = get_dataset()  # tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter
    next(train_dataset.__iter__())