import csv
import os

from os import walk

import numpy as np

path = 'Data_/instances_mosel'
filenames = sorted(next(walk(path), (None, None, []))[2])

for file in filenames:
    with open('Data_/instances_mosel/' + file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        line_count = 0
        mat = []
        for row in csv_reader:
            if 3 <= line_count:
                # print(row)
                mat.append(row[:-1])
            line_count += 1
        mat = np.array(mat[:-1], dtype=np.float64)
        np.savetxt('Data_/csv_/' + file, np.asarray(mat), delimiter=",")