import csv
import pandas as pd
import numpy as np

file = []

with open('Data_/csv_/Dist/Dist/Covid19-418.txt', 'r', newline='\n') as csvfile:
    seq_file = csv.reader(csvfile, delimiter=',', quotechar="'")
    line = 0
    for row in seq_file:
        file.append(row)
        print(row)


names = file[0]
sequences = [el[0] for el in file[1:]]

df = pd.DataFrame({"taxon": names, "sequence": sequences})
df.to_csv('Covid19_sequences.csv', index_label=False, index=False)