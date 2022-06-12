import copy
from os import walk

import numpy as np

from Instances.instance import Instance

path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

D = mats[0]

m = 5

D = D[:m, :m]


pb_0 = Instance(D)
