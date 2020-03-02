"""
References
 - https://github.com/codertimo/BERT-pytorch
 - http://freesearch.pe.kr/archives/4963

Sample data
 - Yelp reviews Polarity from below page
    https://course.fast.ai/datasets
    References
        https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
 - Naver movie review
    https://github.com/e9t/nsmc/
Steps
 0. follow all steps of __main__.py
     1. dataset > vocab.py > build()


"""
import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import argparse
from bert_codertimo_pytorch.dataset.vocab import WordVocab
from bert_codertimo_pytorch.dataset import BERTDataset
from bert_codertimo_pytorch.model import BERT
from bert_codertimo_pytorch.trainer import BERTTrainer

"""
Data pre-processing
"""
data_base_path = 'dataset/yelp_review_polarity_csv/'
test_data_file = data_base_path + 'test.csv'  # 38000
train_data_file = data_base_path + 'train.csv'  # 560000
test_bt_file = test_data_file.replace('.csv', '_bt.csv')
train_bt_file = train_data_file.replace('.csv', '_bt.csv')


def save_df_to_bt_format(input_file_path, output_file_path, top=None):
    """
    >>> save_df_to_bt_format(test_data_file, test_bt_file, top=100)
    >>> save_df_to_bt_format(train_data_file, train_bt_file, top=1000)
    """
    df = pd.read_csv(input_file_path, header=None, names=['label', 'txt'])

    sent_pairs = ['\t'.join(line_split[:2])
                  for line_split in df.txt.map(lambda x: x[:-1].split('. ')) if len(line_split) >= 2]
    bt = pd.DataFrame({'txt': sent_pairs})
    if top:
        bt = bt.loc[:top, :]
    else:
        pass

    bt.to_csv(output_file_path, header=False, index=False, encoding='utf-8')


# df_train = pd.read_csv(train_data_file, encoding='utf-8').loc[:1000, :]
# df_test = pd.read_csv(test_data_file, encoding='utf-8').loc[:200, :]
bt_train = pd.read_csv(train_bt_file, encoding='utf-8')
bt_train.shape


class Args:
    def __init__(self):
        pass


args = Args()
# Data source
args.train_dataset = data_base_path + train_bt_file
args.test_dataset = data_base_path + test_bt_file
args.corpus_path = train_bt_file  # 'bert_pytorch/dataset/sample_corpus.txt'
args.vocab_path = data_base_path + 'vocab.pkl'
args.output_path = data_base_path + 'bert_trained.model'
args.encoding = 'utf-8'

# Model settings
args.hidden = 512
args.layers = 2  # number of transformer_annotated blocks
args.attn_heads = 4
args.seq_len = 64

# Data feed options
args.batch_size = 256
args.epochs = 30
args.num_workers = 0
args.vocab_size = 30000
args.min_freq = 1  # ???
args.corpus_lines = bt_train.shape[0]
args.on_memory = True

# Training settings
args.lr = 1e-3
args.adam_beta1 = 0.9
args.adam_beta2 = 0.999
args.adam_weight_decay = 0.01

args.with_cuda = True
args.log_freq = 10  # printing loss every n iter: setting n
args.cuda_devices = 0

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
>>> BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines, on_memory=args.on_memory)

about Dataset class
 - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
train_dataset = BERTDataset(corpus_path=train_bt_file, vocab=vocab, seq_len=args.seq_len,
                            corpus_lines=args.corpus_lines, on_memory=True)
test_dataset = BERTDataset(corpus_path=test_bt_file, vocab=vocab, seq_len=args.seq_len, on_memory=args.on_memory)

"""
DataLoader
"""
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

"""
Build model
"""
bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

"""
Trainer
"""
trainer = BERTTrainer(
    bert=bert, vocab_size=len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
    lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
    with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

"""
Train
"""
for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)

    if test_data_loader is not None:
        trainer.test(epoch)
