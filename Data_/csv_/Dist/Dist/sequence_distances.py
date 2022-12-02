import copy
import csv
import time

import pandas as pd
import numpy as np
import scipy
from Bio.Seq import Seq
import dask.dataframe as dd

from pyspark.sql.types import *
import pyspark


def compute_alignment(seq1, seq2, k):
    seq_1 = Seq(seq1)
    seq_2 = Seq(seq2)

    min_seq_size = min([len(seq_1), len(seq_2)])
    seq1, seq2 = (seq_1, seq_2) if len(seq1) < len(seq2) else (seq_2, seq_1)
    words = {}
    for i in range(min_seq_size - k):
        word = seq1[i: i + k]
        if word not in words.keys():
            words[word] = [seq1.count_overlap(word), seq2.count_overlap(word)]
        word = seq2[i: i + k]
        if word not in words.keys():
            words[word] = [seq1.count_overlap(word), seq2.count_overlap(word)]

    n_words_seq_1 = len(words)

    # remaining words from longer seq
    if len(seq1) != len(seq2):
        for i in range(min_seq_size - k, len(seq2) - k):
            word = seq2[i: i + k]
            if word not in words.keys():
                words[word] = [0, seq2.count_overlap(word)]

    n_words_seq_2 = len(words)

    counts = np.array(list(words.values()))
    frequencies = counts.astype(np.float64)
    frequencies[:, 0] = frequencies[:, 0] / n_words_seq_1
    frequencies[:, 1] = frequencies[:, 1] / n_words_seq_2

    return words, counts, frequencies


def euclidian(counts):
    return np.sum((counts[:, 0] - counts[:, 1]) ** 2)


def KL(freq):
    return np.sum(freq[:, 0] * np.log2((freq[:, 0] + 1) / (freq[:, 1] + 1))), \
           np.sum(freq[:, 1] * np.log2((freq[:, 1] + 1) / (freq[:, 0] + 1)))


def cos_evol(counts):
    cos = np.dot(counts[:, 0], counts[:, 1]) / (np.linalg.norm(counts[:, 0]) * np.linalg.norm(counts[:, 0]))
    evol = -np.log((1 + cos) / 2)
    return cos, evol


def count_words(seq, word):
    seq_ = Seq(seq)
    return seq_.count_overlap(word)


def get_words(seq_str, k):
    words = set()
    for i in range(len(seq_str) - k):
        words.add(seq_str[i: i + k])

    return words


df = pd.read_csv("Data_/csv_/Dist/Dist/Covid19_sequences.csv")
ddf = dd.from_pandas(df, npartitions=4)
dask_series = ddf['sequence'].apply(get_words, args=(8,), meta=('x', 'int'))
ddf['word_set'] = dask_series

dff = ddf.compute()
all_words_set = set().union(*dff.word_set.to_list())
#
# ddf = dd.from_pandas(dff, npartitions=4)
# for i, word in enumerate(all_words_set):
#     print(i)
#     dask_series = ddf['sequence'].apply(count_words, args=(word,), axis=1, meta=pd.Series(dtype="int"))
#     ddf[word] = dask_series
#
# final_df = ddf.compute()

sc
