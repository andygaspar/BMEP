import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
        else:
            idx = np.array(np.where(np.triu(s[1:, 1:]) == 1)).T[0] + 1
            i, j = idx
            k += 1
            G.add_nodes_from([(s[0, j], {"color": "red"})])
            G.add_edges_from([(s[0, i], s[0, j])])
            over = True
    print(G.nodes)
    mapping = dict(zip(G, [labels[int(node)] if int(node) < n else int(node) for node in G]))
    G = nx.relabel_nodes(G, mapping)

    return G
