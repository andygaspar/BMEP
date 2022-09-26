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
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

# random.seed(0)

instance = 2 # m17

num_instances = 20_000
d_mat_initial = mats[instance]
dim_min = 5
dim_max = 9
max_time = 20
total_time_h = 12
total_time = total_time_h * 60 * 60
name_folder = filenames[instance][:-4]
gen = Generator(name_folder=name_folder, num_instances=num_instances, d_mat_initial=d_mat_initial, dim_min=dim_min,
                dim_max=dim_max, max_time=max_time, total_time=total_time, pardi_solver=True)


