import pandas as pd

import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import argparse
from bert_pytorch.dataset.vocab import WordVocab
from bert_pytorch.dataset import BERTDataset
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer


class Args:
    def __init__(self):
        pass


args = Args()
# Data source


data_base_path = 'dataset/naver_movie_review/'
train_data_file = 'naver_movie_review.txt'

args.corpus_path = data_base_path + train_data_file
args.encoding = 'utf-8'
args.vocab_size = 30000
args.min_freq = 1


"""
Building vocab
>>> vocab = WordVocab.load_vocab(args.vocab_path)
"""
with open(args.corpus_path, 'r', encoding=args.encoding) as f:
    vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

df = pd.read_csv(data_base_path + train_data_file, sep='\t')
df.columns
df.head()
