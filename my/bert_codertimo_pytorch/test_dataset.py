from bert_pytorch.dataset import CustomBERTDataset
from bert_pytorch.dataset import WordVocab


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
args.layers = 2  # number of transformer_annotated blocks
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


def test_custom_dataset():
    vocab = WordVocab.load_vocab(args.vocab_path)

    cd = CustomBERTDataset(
        corpus_path=args.corpus_path,
        vocab=vocab,
        seq_len=args.seq_len,
        encoding=args.encoding,
        corpus_lines=args.corpus_lines,
        on_memory=args.on_memory
    )

    t1 = cd.get_random_line()
    t1_random, t1_label = cd.random_word(t1)
    t1_random
    t1_label

    cd[0]
