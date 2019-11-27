# -*- coding: utf-8 -*-
"""
Created at 2019-11-26

@author: dongwan.kim

References
 - https://github.com/kpe/bert-for-tf2
"""
import numpy as np
import random


def parity_ds_generator(batch_size=32, max_len=10, max_int=4, modulus=2):
    """
    Generates a parity calculation dataset (seq -> sum(seq) mod 2),
    where seq is a sequence of length less than max_len
    of integers in [1..max_int).
    """
    while True:
        data = np.zeros((batch_size, max_len))
        tag = np.zeros(batch_size, dtype='int32')
        for i in range(batch_size):
            datum_len = random.randint(1, max_len - 1)
            total = 0
            for j in range(datum_len):
                data[i, j] = random.randint(1, max_int)
                total += data[i, j]
            tag[i] = total % modulus
        yield data, tag  # ([batch_size, max_len], [max_len])


g = parity_ds_generator()
next(g)

13 % 5