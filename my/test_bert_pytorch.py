"""
References
 - https://github.com/codertimo/BERT-pytorch
 - http://freesearch.pe.kr/archives/4963

Sample data
 - Yelp reviews Polarity from below page
    https://course.fast.ai/datasets
    References
        https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04

Steps
 0. follow all steps of __main__.py
     1. dataset > vocab.py > build()


"""
import tqdm
import pandas as pd
import argparse
from bert_pytorch.dataset.vocab import WordVocab
from bert_pytorch.dataset import BERTDataset


"""
Data pre-processing
"""
test_data_file = 'dataset/yelp_review_polarity_csv/test.csv'  # 38000
train_data_file = 'dataset/yelp_review_polarity_csv/train.csv'  # 560000
test_bt_file = test_data_file.replace('.csv', '_bt.csv')
train_bt_file = train_data_file.replace('.csv', '_bt.csv')


def save_df_to_bt_format(input_file_path, output_file_path):
    """
    >>> save_df_to_bt_format(test_data_file, test_bt_file)
    >>> save_df_to_bt_format(train_data_file, train_bt_file)
    """
    df = pd.read_csv(input_file_path, header=None, names=['label', 'txt'])
    sent_pairs = ['\t'.join(line_split[:2])
                  for line_split in df.txt.map(lambda x: x[:-1].split('. ')) if len(line_split) >= 2]
    bt = pd.DataFrame({'txt': sent_pairs})
    bt.to_csv(output_file_path, header=False, index=False, encoding='utf-8')


class Args:
    def __init__(self):
        pass


args = Args()
args.corpus_path = 'bert_pytorch/dataset/sample_corpus.txt'
args.output_path = 'bert_pytorch/dataset/sample_vocab.pkl'
args.encoding = 'utf-8'
args.vocab_size = 10
args.min_freq = 1
args.train_dataset = None
args.seq_len = None
args.corpus_lines = None
args.on_memory = None
args.test_dataset = None
args.batch_size = None
args.num_workers = None

"""
Building vocab
>>> vocab = WordVocab.load_vocab(args.vocab_path)
"""
with open(args.corpus_path, 'r', encoding=args.encoding) as f:
    vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

print('Vocab size:', len(vocab))
vocab.save_vocab(args.output_path)


with open(test_bt_file, 'r', encoding=args.encoding) as f:
    vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)



"""
Bert dataset
>>> BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines, on_memory=args.on_memory)
"""
bert_data = BERTDataset(corpus_path=test_bt_file, vocab=vocab, seq_len=64, corpus_lines=1000, on_memory=True)
bert_data[0]

corpus_lines = 0
with open(test_bt_file, "r", encoding='utf-8') as f:
    lines = [line[:-1].split("\t")
             for line in tqdm.tqdm(f, desc="Loading Dataset", total=38000)]
    corpus_lines = len(lines)

    for line in tqdm.tqdm(f, desc="Loading Dataset", total=560000):
        corpus_lines += 1
        print(line)
        print(type(line))
        print(line[:-1])
        if corpus_lines > 10:
            break

ts = "Wow	What a shame"
ts[:-1].split('\t')[:1]
ts.split('\t')


