import csv
import time

import pandas as pd
import numpy as np
import scipy
from Bio.Seq import Seq


df = pd.read_csv("Data_/csv_/Dist/Dist/Covid19_sequences.csv")

# import dask.dataframe as dd
#
# ddf = dd.from_pandas(df, npartitions=4)
# dask_series = ddf.apply(foo, axis=1, args=(10,), x=100, meta='float')
# ddf['B'] = dask_series

# Convert Dask DataFrame back to Pandas DataFrame
# df = ddf.compute()

virus1 = df.sequence[0]
virus2 = df.sequence[1]
seq1 = Seq(virus1)
seq2 = Seq(virus2)

min_seq_size = min([len(seq1), len(seq2)])
t = time.time()
k = 8
words = {}
for i in range(min_seq_size - k):
    word = seq1[i: i + k]
    if word not in words.keys():
        words[word] = [seq1.count_overlap(word), seq2.count_overlap(word)]
    word = seq2[i: i + k]
    if word not in words.keys():
        words[word] = [seq1.count_overlap(word), seq2.count_overlap(word)]

if len(seq1) != len(seq2):
    if len(seq1) < len(seq2):
        for i in range(min_seq_size - k, len(seq2) - k):
            word = seq2[i: i + k]
            if word not in words.keys():
                words[word] = [0, seq2.count_overlap(word)]
    else:
        for i in range(min_seq_size - k, len(seq1) - k):
            word = seq1[i: i + k]
            if word not in words.keys():
                words[word] = [seq1.count_overlap(word), 0]

counts = np.array(list(words.values()))
print(time.time() - t)
np.linalg.norm(counts[:, 0] - counts[:, 1])

scipy.special.comb(418, 2, exact=True) * 2.5 / 60**2
60/16