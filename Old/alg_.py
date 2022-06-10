import copy
import string

import networkx as nx
from matplotlib import pyplot as plt

from print_graph import get_graph
import numpy as np


def get_pardi(g, labels):
    m = (len(g) + 2) // 2
    n = len(g)
    nodes = labels + [i for i in range(m, 2 * m - 2)]
    node_dict = dict(zip(nodes, range(len(nodes))))
    node_dict_back = dict(zip(range(len(nodes)), nodes))

    ad_mat = np.zeros((len(nodes), len(nodes)))

    for edge in g.edges:
        a, b = edge
        ad_mat[node_dict[a], node_dict[b]] = 1
        ad_mat[node_dict[b], node_dict[a]] = 1

    ad_mat_1 = copy.deepcopy(ad_mat)

    leafs = nodes[:m]
    leafs = leafs[::-1]
    internal_nodes = {m: m}
    pardi = {}
    for k, l in enumerate(leafs[:-3]):
        x = node_dict[l]
        j = np.where(ad_mat_1[x] == 1)[0][-1]
        internal_nodes[j] = n - 1 - k
        ad_mat_1[j, x] = 0
        i = np.where(ad_mat_1[j] == 1)[0][0]
        if i < m:
            print(l, "->", node_dict_back[i])
            pardi[l] = [node_dict_back[i]]
        else:
            ii = np.where(ad_mat_1[j] == 1)[0][1]
            print(l, "->", node_dict_back[i], node_dict[ii])
            pardi[l] = [node_dict_back[i], node_dict_back[ii]]
        k = np.where(ad_mat_1[j] == 1)[0][-1]
        ad_mat_1[k, j] = 0
        ad_mat_1[k, i] = 1
        ad_mat_1[i, k] = 1
        ad_mat_1[i, j] = 0
        ad_mat_1[x] = 0
        ad_mat_1[:, x] = 0
        ad_mat_1[j] = 0
        ad_mat_1[:, j] = 0
        # g = nx.from_numpy_matrix(ad_mat_1)
        # mapping = dict(zip(g, nodes))
        # g = nx.relabel_nodes(g, mapping)
        # nx.draw(g, node_color=["red" if i < m else "green" for i in range(2 * m - 2)],
        #         with_labels=True, font_weight='bold')
        # plt.show()

    print(internal_nodes)
    # map_nodes = dict(zip(list(internal_nodes.values())[::-1], range(m + 1, 2*m - 2)))
    # mapping = dict(zip(g, [node if node not in internal_nodes else map_nodes[node] for node in g.nodes]))
    # g = nx.relabel_nodes(g, mapping)
    # nx.draw(g, node_color=["red" if i < m else "green" for i in range(2 * m - 2)],
    #         with_labels=True, font_weight='bold')
    # plt.show()

    pardi = dict(zip(list(pardi.keys())[::-1], list(pardi.values())[::-1]))
    for key in pardi.keys():
        insertion = pardi[key]
        if len(insertion) == 2:
            pardi[key] = [internal_nodes[pardi[key][0]], internal_nodes[pardi[key][1]]]
    print(pardi)
    return pardi, g




