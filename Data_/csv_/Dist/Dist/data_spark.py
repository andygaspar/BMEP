import pandas as pd
import dask.dataframe as dd

from Bio.Seq import Seq
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, IntegerType


def get_words(seq_str, k):
    words = set()
    for i in range(len(seq_str) - k):
        words.add(seq_str[i: i + k])

    return words

@udf(IntegerType())
def count_words(seq, word):
    seq_ = Seq(seq)
    return seq_.count_overlap(word)

spark = SparkSession.builder.getOrCreate()

df_pandas = pd.read_csv("Data_/csv_/Dist/Dist/Covid19_sequences.csv")
ddf_pandas = dd.from_pandas(df_pandas, npartitions=4)
dask_series = ddf_pandas['sequence'].apply(get_words, args=(8,), meta=('x', 'int'))
ddf_pandas['word_set'] = dask_series

df_pandasf = ddf_pandas.compute()
all_words_set = set().union(*df_pandasf.word_set.to_list())



df = spark.read.csv('Data_/csv_/Dist/Dist/Covid19_sequences.csv', inferSchema=True, header=True)
# df = df.withColumn('word_set', get_words(f.col('sequence'), f.lit(8)))

print("here *************************************")
i = 0
for word in list(all_words_set)[:1000]:
    df = df.withColumn(word, count_words(f.col('sequence'), f.lit(word)))
    i += 1
df.show()