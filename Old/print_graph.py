import copy
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Solvers.SWA.swa_solver import SwaSolver


def get_graph(matrix, labels=None):
    if labels is None:
        labels = [range(matrix.shape[0])]
    G = nx.Graph()
    s = copy.deepcopy(matrix)
    s = np.vstack([range(s.shape[0]), s])
    col = np.array([0] + [i for i in range(s.shape[0] - 1)])
    s = np.c_[col, s]
    over = False
    k = 0
    n = s.shape[0] - 1

    while not over:
        if s.shape[0] > 3:
            idx = np.array(np.where(np.triu(s[1:, 1:]) == 2)).T[0] + 1
            i, j = idx
            if s[0, i] < n and s[0, j] < n:
                G.add_nodes_from([(s[0, i], {"color": "red"})])
                G.add_nodes_from([(s[0, j], {"color": "red"})])

                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(s[0, i], n + k), (s[0, j], n + k)])

            elif s[0, j] < n:
                G.add_nodes_from([(s[0, j], {"color": "red"})])
                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(s[0, i], n + k), (s[0, j], n + k)])

            else:
                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(s[0, i], n + k), (s[0, j], n + k)])

            i, j = idx
            s[i, 0] = n + k
            s[0, i] = n + k
            s[i, 1:] -= 1
            s[1:, i] -= 1
            k += 1
            s = np.delete(s, j, axis=0)
            s = np.delete(s, j, axis=1)

            nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
                    with_labels=True, font_weight='bold')
            plt.show()
        else:
            idx = np.array(np.where(np.triu(s[1:, 1:]) == 1)).T[0] + 1
            i, j = idx
            k += 1
            G.add_nodes_from([(s[0, j], {"color": "red"})])
            G.add_edges_from([(s[0, i], s[0, j])])
            over = True

    mapping = dict(zip(G, [labels[int(node)] if int(node) < n else int(node) for node in G]))
    G = nx.relabel_nodes(G, mapping)

    return G


T = np.array([[0, 3, 5, 5, 2, 4],
              [3, 0, 3, 4, 4, 3],
              [5, 3, 0, 2, 5, 4],
              [5, 4, 2, 0, 5, 3],
              [2, 4, 5, 5, 0, 3],
              [4, 3, 4, 3, 3, 0]])

get_graph(T, ["A", "B", "C", "D", "E", "F"])

import os
import numpy as np

path = 'Data_/csv_'
filenames = sorted(next(os.walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

mat = mats[2][:7, :7]/np.max(mats[2][:7, :7])
mat_sum = mat.sum(axis=1)
mat_means = mat.mean(axis=1)
mat_stad = mat.std(axis=1)
swa = SwaSolver(mat[:5, :5])
swa.solve()
tau_full = swa.get_tau(swa.solution)
tau = tau_full[:5, :5]


def get_taxa_standing(d, tau, taxa_idx):
    return sum(d[taxa_idx] / 2 ** (tau[taxa_idx]))


def get_internal_standing(d_means, tau_full, internal_idx):
    return sum(d_means / 2 ** tau_full[internal_idx, :d_means.shape[0]])


for i in range(5):
    print(get_taxa_standing(mat[:5, :5], tau, i))

for j in range(5, 8):
    print(get_internal_standing(mat_means, tau_full, j))
