import torch.nn as nn

from .bert import CustomBERT


class GoodBad(nn.Module):
    def __init__(self, bert: CustomBERT):
        super().__init__()
        self.bert = bert

    def forward(self, x):
        return self.bert(x)
