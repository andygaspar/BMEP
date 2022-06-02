import copy
import string
from os import walk

import networkx as nx
import numpy as np
import gurobipy as gb
from itertools import combinations

from instance import Instance

path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

D = mats[0]

m = 6

D = D[:m, :m]


pb_0 = Instance(D)
print(pb_0.T)
print(pb_0.obj_val)

solution_xp = np.array([[0, 2, 4, 5, 5, 3],
                        [2, 0, 4, 5, 5, 3],
                        [4, 4, 0, 3, 3, 3],
                        [5, 5, 3, 0, 2, 4],
                        [5, 5, 3, 2, 0, 4],
                        [3, 3, 3, 4, 4, 0]])

M = range(m)
print(sum(2 ** float(-solution_xp[i, j]) * D[i, j] for j in M for i in M if i != j))

d = copy.deepcopy(D[:m, :m])

dist_sum = np.sum(d, axis=0)
order = np.argsort(dist_sum)
sorted_d = np.zeros_like(d)
for i in order:
    for j in order:
        sorted_d[i, j] = d[order[i], order[j]]



pb_1 = Instance(sorted_d)
print(pb_1.T)
print(pb_1.obj_val)

pb_0.print_graph()
pb_1.print_graph()

