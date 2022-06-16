import copy
import random
import warnings
from os import walk

import numpy as np

from Instances.generator import Generator
from Instances.instance import Instance

warnings.simplefilter("ignore")

path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

# random.seed(0)

num_instances, d_mat_initial, dim, max_time, total_time = 1000, mats[3], 6, 6, 3600 * 8
gen = Generator(num_instances, d_mat_initial, dim, max_time=max_time, total_time=total_time)


