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

import string
pb_1.graph.edges
nodes = [s for s in string.ascii_uppercase[:6]] + [i for i in range(6,10)]
node_dict = dict(zip(nodes, range(len(nodes))))

ad_mat = np.zeros((len(nodes), len(nodes)))
for edge in pb_1.graph.edges:
    a, b = edge
    ad_mat[node_dict[a], node_dict[b]] = 1
    ad_mat[node_dict[b], node_dict[a]] = 1

ad_mat_1 = copy.deepcopy(ad_mat)

leafs = nodes[:6]
leafs = leafs[::-1]

for l in leafs:
    j = np.where(ad_mat_1[nodenode])

[[0 3 2 5 4 5]
 [3 0 3 4 3 4]
 [2 3 0 5 4 5]
 [5 4 5 0 3 2]
 [4 3 4 3 0 3]
 [5 4 5 2 3 0]]