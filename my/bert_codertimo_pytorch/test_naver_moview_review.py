import re

from datetime import datetime
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split

from bert_codertimo_pytorch.dataset.vocab import WordVocab
from bert_codertimo_pytorch.dataset import BERTDataset, CustomBERTDataset
from bert_codertimo_pytorch.model import BERT, CustomBERT, CustomBERTLM, GoodBad
from bert_codertimo_pytorch.trainer import BERTTrainer, CustomBERTTrainer


class Args:
    def __init__(self):
        pass


args = Args()
# Data source


data_base_path = 'dataset/naver_movie_review/'
raw_data_file = 'ratings.txt'
train_bt_file = 'corpus_train.txt'
test_bt_file = 'corpus_test.txt'

args.raw_dataset = data_base_path + raw_data_file
args.train_dataset = data_base_path + train_bt_file
args.test_dataset = data_base_path + test_bt_file
args.corpus_path = data_base_path + 'corpus.txt'
args.vocab_path = data_base_path + 'vocab.pkl'
args.output_path = data_base_path + 'bert_trained.model'
args.encoding = 'utf-8'

# Model settings
args.hidden = 512
args.layers = 2  # number of transformer blocks
args.attn_heads = 4
args.seq_len = 64

# Data feed options
args.batch_size = 256
args.epochs = 3
args.num_workers = 0
args.vocab_size = 30000
args.min_freq = 1  # ???
args.corpus_lines = 200000
args.on_memory = True
args.test_dataset_ratio = 0.1

# Training settings
args.lr = 1e-3
args.adam_beta1 = 0.9
args.adam_beta2 = 0.999
args.adam_weight_decay = 0.01

args.with_cuda = True
args.log_freq = 10  # printing loss every n iter: setting n
args.cuda_devices = [2]


# 초성 리스트. 00 ~ 18
cho = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
jung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
jong = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ',
        'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


# pre-process
def split_ko_by_jamo(sentence, sep=' '):
    """
    References
        https://frhyme.github.io/python/python_korean_englished/

    한글 단어를 입력받아서 초성/중성/종성을 구분하여 리턴해줍니다.

    >>> split_ko_by_jamo(' 안녕 ㅎㅇ')
    """
    global cho, jung, jong
    sentence = re.sub('[ ]', '', sentence)
    r_lst = ''

    for w in sentence:
        if '가' <= w <= '힣':
            ch1 = (ord(w) - ord('가')) // 588
            ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2
            r_lst += sep.join([cho[ch1], jung[ch2], jong[ch3]]) + sep
        elif w == sep:
            r_lst += w
        else:
            r_lst += w + sep

        r_lst += '[EOC]' + sep

    return r_lst


def split_ko_by_char(sentence, sep=' '):
    """
    split_ko_by_char('안녕 하hello')
    """
    r_lst = ''
    for w in sentence:
        if '가' <= w <= '힣':
            r_lst += (w + sep)
        else:
            r_lst += w

    return r_lst


df_raw = pd.read_csv(args.raw_dataset, sep='\t', encoding='utf-8')
df_raw['document_cleaned'] = df_raw['document'].astype(str).map(split_ko_by_jamo)
# df_raw[['document_cleaned']].to_csv(args.corpus_path, index=False, header=False)
df_train, df_test = train_test_split(df_raw[['document_cleaned']], test_size=args.test_dataset_ratio, random_state=0)
# df_train.to_csv(args.train_dataset, index=False, header=False)
# df_test.to_csv(args.test_dataset, index=False, header=False)
args.train_dataset_lines = df_train.shape[0]
args.test_dataset_lines = df_test.shape[0]


"""
Building vocab
>>> vocab = WordVocab.load_vocab(args.vocab_path)
"""
# vocab = WordVocab(df_raw['document_cleaned'].values, max_size=args.vocab_size, min_freq=args.min_freq)
# vocab.save_vocab(args.vocab_path)
vocab = WordVocab.load_vocab(args.vocab_path)
print('Vocab size:', len(vocab))
vocab.freqs

"""
Bert dataset without NSP
"""
train_dataset = CustomBERTDataset(corpus_path=args.train_dataset, vocab=vocab, seq_len=args.seq_len,
                                  corpus_lines=args.train_dataset_lines, on_memory=True)
test_dataset = CustomBERTDataset(corpus_path=args.test_dataset, vocab=vocab, seq_len=args.seq_len,
                                 corpus_lines=args.test_dataset_lines, on_memory=True)

"""
DataLoader
"""
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

"""
Build model
"""
custom_bert = CustomBERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

"""
Trainer
"""
trainer = CustomBERTTrainer(
    bert=custom_bert, vocab_size=len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
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


"""
etc
"""
saved_model = torch.load(args.output_path + '.ep1')
saved_model.eval()
type(saved_model)

custom_model = GoodBad(
    bert=CustomBERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
)

custom_model.bert = saved_model


custom_model.forward()