import pandas as pd

import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import argparse
from bert_pytorch.dataset.vocab import WordVocab
from bert_pytorch.dataset import BERTDataset
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer
import re


class Args:
    def __init__(self):
        pass


args = Args()
# Data source


data_base_path = 'dataset/naver_movie_review/'
raw_data_file = 'naver_movie_review.txt'
train_bt_file = ''
test_bt_file = ''

args.train_dataset = data_base_path + train_bt_file
args.test_dataset = data_base_path + test_bt_file
args.corpus_path = data_base_path + raw_data_file  # 'bert_pytorch/dataset/sample_corpus.txt'
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
args.epochs = 30
args.num_workers = 0
args.vocab_size = 30000
args.min_freq = 1  # ???
args.corpus_lines = 200000
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
def korean_to_be_englished(korean_word):
    """
    References
        https://frhyme.github.io/python/python_korean_englished/

    한글 단어를 입력받아서 초성/중성/종성을 구분하여 리턴해줍니다.

    >>> korean_to_be_englished('아녕')
    """
    ####################################
    # 초성 리스트. 00 ~ 18
    cho = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    # 중성 리스트. 00 ~ 20
    jung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    jong = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    ####################################

    korean_word = re.sub('[ ]', '', korean_word)
    r_lst = ''

    for w in list(korean_word.strip()):
        if '가'<=w<='힣':
            ch1 = (ord(w) - ord('가'))//588
            ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
            print(ch1, ch2, ch3, type(ch3))
            r_lst += ' '.join([cho[ch1], jung[ch2], jong[ch3]]) + ' '
        else:
            r_lst += w + ' '

        r_lst += '[EOC] '

    return r_lst

df_raw = pd.read_csv(args.corpus_path, sep='\t', encoding='utf-8')
df_raw['document'] = df_raw['document'].astype(str).map(korean_to_be_englished)
df_raw['document'][0]
# df_raw.document.values
# with open(args.corpus_path, 'r', encoding=args.encoding) as f:
#     vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)
vocab = WordVocab(df_raw.document.values, max_size=args.vocab_size, min_freq=args.min_freq)
vocab.save_vocab(args.output_path)

'abc'.