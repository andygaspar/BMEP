import copy
import string

import networkx as nx
from matplotlib import pyplot as plt

from print_graph import get_graph
import numpy as np

T = np.array([[0, 3, 2, 5, 4, 5],
              [3, 0, 3, 4, 3, 4],
              [2, 3, 0, 5, 4, 5],
              [5, 4, 5, 0, 3, 2],
              [4, 3, 4, 3, 0, 3],
              [5, 4, 5, 2, 3, 0]])

T = np.array([[0, 4, 2, 4, 4, 4],
              [4, 0, 4, 4, 4, 2],
              [2, 4, 0, 4, 4, 4],
              [4, 4, 4, 0, 2, 4],
              [4, 4, 4, 2, 0, 4],
              [4, 2, 4, 4, 4, 0]])

T = np.array([[0, 4, 2, 5, 5, 4, 4],
              [4, 0, 4, 5, 5, 2, 4],
              [2, 4, 0, 5, 5, 4, 4],
              [5, 5, 5, 0, 2, 5, 3],
              [5, 5, 5, 2, 0, 5, 3],
              [4, 2, 4, 5, 5, 0, 4],
              [4, 4, 4, 3, 3, 4, 0]])

# T = np.array([[0, 4, 2, 4, 3],
#               [4, 0, 4, 2, 3],
#               [2, 4, 0, 4, 3],
#               [4, 2, 4, 0, 3],
#               [3, 3, 3, 3, 0]])

m = T.shape[0]
labels = [s for s in string.ascii_uppercase[:m]]
G = get_graph(T, labels)
nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
        with_labels=True, font_weight='bold')
plt.show()

nodes = labels + [i for i in range(m, 2 * m - 2)]
node_dict = dict(zip(nodes, range(len(nodes))))
node_dict_back = dict(zip(range(len(nodes)), nodes))

ad_mat = np.zeros((len(nodes), len(nodes)))

for edge in G.edges:
    a, b = edge
    ad_mat[node_dict[a], node_dict[b]] = 1
    ad_mat[node_dict[b], node_dict[a]] = 1

ad_mat_1 = copy.deepcopy(ad_mat)

leafs = nodes[:m]
leafs = leafs[::-1]

for l in leafs[:-3]:
    x = node_dict[l]
    j = np.where(ad_mat_1[x] == 1)[0][-1]
    ad_mat_1[j, x] = 0
    i = np.where(ad_mat_1[j] == 1)[0][0]
    print(l, "->", node_dict_back[i])
    k = np.where(ad_mat_1[j] == 1)[0][-1]
    ad_mat_1[k, j] = 0
    ad_mat_1[k, i] = 1
    ad_mat_1[i, k] = 1
    ad_mat_1[i, j] = 0
    ad_mat_1[x] = 0
    ad_mat_1[:, x] = 0
    ad_mat_1[j] = 0
    ad_mat_1[:, j] = 0
    g = nx.from_numpy_matrix(ad_mat_1)
    mapping = dict(zip(g, nodes))
    g = nx.relabel_nodes(g, mapping)
    nx.draw(g, node_color=["red" if i < m else "green" for i in range(2 * m - 2)],
            with_labels=True, font_weight='bold')
    plt.show()
