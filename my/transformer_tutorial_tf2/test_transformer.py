from transformer_tutorial_tf2.transformer import *


def test_get_dataset():
    train_dataset, val_examples = get_dataset()
    print(train_dataset, val_examples)


def test_get_dataset():
    train_dataset, val_examples = get_dataset()
    tokenizer_en, tokenizer_pt = get_tokenizer(train_dataset)

    sample_string = 'Transformer is awesome.'

    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


def test_encode():
    sample_string = 'Transformer is awesome.'

    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string


def test_get_dataset():
    train_dataset, val_dataset = get_dataset()
    pt_batch, en_batch = next(iter(val_dataset))
