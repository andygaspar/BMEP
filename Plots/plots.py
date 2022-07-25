import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_phylogeny(leaf_edges, internal_edges, labels=None, size=1000, filename=None, show=True):
    leafs = ["a_" + str(k) for k in range(1, len(leaf_edges) + 1)] if labels is None else labels
    print(leafs)
    edges = [(leaf, leaf_edges[i]) for i, leaf in enumerate(leafs)] + internal_edges
    print(edges)
    g = nx.Graph()
    g.add_nodes_from([(leaf, {"color": "green"}) for leaf in leafs])
    g.add_nodes_from([(k, {"color": "red"}) for k in range(1, len(leaf_edges) - 1)])

    g.add_edges_from(edges)
    nx.draw(g, node_color=[g.nodes[node]["color"] for node in g.nodes],
            with_labels=True, font_weight='bold', node_size=size)
    if show:
        plt.show()
    if filename is not None:
        plt.savefig("Plots/" + filename)


def plot_pardi(sequence, labels=None, size=1000, filename=None, show=True):
    leafs = ["a_" + str(k) for k in range(1, len(sequence) + 4)] if labels is None else labels
    f_name = filename + '_1' if filename is not None else None
    plot_phylogeny([1, 1, 1], [], leafs[:3], size, filename, show)
    for i, s in enumerate(sequence):
        f_name = filename + '_' + str(i + 2) if filename is not None else None
        pass


l_edges = [1, 1, 2, 2, 4, 4]
i_edges = [(1, 3), (3, 2), (4, 3)]

plot_phylogeny(l_edges, i_edges)

