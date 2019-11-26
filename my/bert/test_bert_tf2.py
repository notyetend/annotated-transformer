# -*- coding: utf-8 -*-
"""
Created at 2019-11-26

@author: dongwan.kim

References
 - https://github.com/kpe/bert-for-tf2
"""
from bert import BertModelLayer


l_bert = BertModelLayer(**BertModelLayer.Params(
    vocab_size=16000,  # embedding params
    use_token_type=True,
    use_position_embeddings=True,
    token_type_vocab_size=2,

    num_layers=12,  # transformer encoder params
    hidden_size=768,
    hidden_dropout=0.1,
    intermediate_size=4 * 768,
    intermediate_activation="gelu",

    adapter_size=None,  # see arXiv:1902.00751 (adapter-BERT)

    shared_layer=False,  # True for ALBERT (arXiv:1909.11942)
    embedding_size=None,  # None for BERT, wordpiece embedding size for ALBERT

    name="bert"  # any other Keras layer params
))
