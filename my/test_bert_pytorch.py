"""
References
 - https://github.com/codertimo/BERT-pytorch


Steps
 0. follow all steps of __main__.py
     1. dataset > vocab.py > build()


"""
import argparse
from bert_pytorch.dataset.vocab import WordVocab


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


"""
Bert dataset
BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)
"""

