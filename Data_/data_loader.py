import random
import warnings
from os import walk
from typing import Union

import numpy as np
warnings.simplefilter("ignore")


class DataSet:

    def __init__(self, d, name):
        self.d = d
        self.name = name
        self.size = self.d.shape[0]

    def __repr__(self):
        return self.name

    def get_full_dataset(self):
        return self.d

    def get_minor(self, *args):
        if len(args) > 1:
            from_, to_ = args
            return self.d[from_: to_, from_: to_]
        else:
            to_ = args[0]
            return self.d[: to_, : to_]

    def generate_random(self, dim, from_=None, to_=None):
        from_ = from_ if from_ is not None else 0
        to_ = to_ if to_ is not None else self.size
        idx = random.sample(range(from_, to_), k=dim)
        return self.d[idx, :][:, idx]


class DistanceData:

    def __init__(self):
        path = 'Data_/csv_'
        filenames = sorted(next(walk(path), (None, None, []))[2])

        self.data_sets = {}
        for file in filenames:
            if file[-4:] == '.txt':
                self.data_sets[file[:-4]] = DataSet(np.genfromtxt('Data_/csv_/' + file, delimiter=','), file[:-4])

    def print_dataset_names(self):
        for key in self.data_sets.keys():
            print(key)

    def get_dataset_names(self):
        return list(self.data_sets.keys())

    def get_dataset(self, data_set: Union[str, int]):
        if type(data_set) == int:
            return list(self.data_sets.values())[data_set]
        else:
            return self.data_sets[data_set]
