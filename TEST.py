import numpy as np
import string

import copy
import string

import networkx as nx
from matplotlib import pyplot as plt

from alg_ import get_pardi
from print_graph import get_graph
import numpy as np

T = np.array([[0, 3, 9, 8, 8, 9, 4, 6, 5, 8, 3, 3],
              [3, 0,10, 9, 9,10, 5, 7, 6, 9, 2, 4],
              [9,10, 0, 5, 5, 2, 7, 5, 6, 3,10, 8],
              [8, 9, 5, 0, 2, 5, 6, 4, 5, 4, 9, 7],
              [8, 9, 5, 2, 0, 5, 6, 4, 5, 4, 9, 7],
              [9,10, 2, 5, 5, 0, 7, 5, 6, 3,10, 8],
              [4, 5, 7, 6, 6, 7, 0, 4, 3, 6, 5, 3],
              [6, 7, 5, 4, 4, 5, 4, 0, 3, 4, 7, 5],
              [5, 6, 6, 5, 5, 6, 3, 3, 0, 5, 6, 4],
              [8, 9, 3, 4, 4, 3, 6, 4, 5, 0, 9, 7],
              [3, 2,10, 9, 9,10, 5, 7, 6, 9, 0, 4],
              [3, 4, 8, 7, 7, 8, 3, 5, 4, 7, 4, 0]])



# T = np.array([[0, 3, 2, 5, 4, 5],
#               [3, 0, 3, 4, 3, 4],
#               [2, 3, 0, 5, 4, 5],
#               [5, 4, 5, 0, 3, 2],
#               [4, 3, 4, 3, 0, 3],
#               [5, 4, 5, 2, 3, 0]])

# T = np.array([[0, 4, 2, 4, 4, 4],
#               [4, 0, 4, 4, 4, 2],
#               [2, 4, 0, 4, 4, 4],
#               [4, 4, 4, 0, 2, 4],
#               [4, 4, 4, 2, 0, 4],
#               [4, 2, 4, 4, 4, 0]])

# T = np.array([[0, 4, 2, 5, 5, 4, 4],
#               [4, 0, 4, 5, 5, 2, 4],
#               [2, 4, 0, 5, 5, 4, 4],
#               [5, 5, 5, 0, 2, 5, 3],
#               [5, 5, 5, 2, 0, 5, 3],
#               [4, 2, 4, 5, 5, 0, 4],
#               [4, 4, 4, 3, 3, 4, 0]])

# T = np.array([[0, 4, 2, 4, 3],
#               [4, 0, 4, 2, 3],
#               [2, 4, 0, 4, 3],
#               [4, 2, 4, 0, 3],
#               [3, 3, 3, 3, 0]])

# T = np.array([[0, 4, 2, 5, 5, 4, 4],
#               [4, 0, 4, 5, 5, 2, 4],
#               [2, 4, 0, 5, 5, 4, 4],
#               [5, 5, 5, 0, 2, 5, 3],
#               [5, 5, 5, 2, 0, 5, 3],
#               [4, 2, 4, 5, 5, 0, 4],
#               [4, 4, 4, 3, 3, 4, 0]])


T = np.array([[0, 5, 2, 4, 5, 3],
              [5, 0, 5, 3, 2, 4],
              [2, 5, 0, 4, 5, 3],
              [4, 3, 4, 0, 3, 3],
              [5, 2, 5, 3, 0, 4],
              [3, 4, 3, 3, 4, 0]])

m = T.shape[0]
labels = [s for s in string.ascii_uppercase[:m]]
G = get_graph(T, labels)
nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
        with_labels=True, font_weight='bold')
plt.show()

sequence, g = get_pardi(G, labels)


def build_pardi(sequence):
    G = nx.Graph()
    m = len(sequence) + 3
    G.add_nodes_from([("A", {"color": "red"}), ("B", {"color": "red"}),
                      ("C", {"color": "red"}), (m, {"color": "green"})])
    G.add_edges_from([("A", m), ("B", m), ("C", m)])
    nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
            with_labels=True, font_weight='bold')
    plt.show()
    for k, node in enumerate(sequence.keys()):
        if len(sequence[node]) < 2:
            internal_node = m + 1 + k
            G.add_nodes_from([(node, {"color": "red"}), (internal_node, {"color": "green"})])
            removing_edge = list(G.edges(sequence[node][0]))[0]
            G.remove_edge(removing_edge[0], removing_edge[1])
            G.add_edges_from([(node, internal_node), (sequence[node][0], internal_node),
                              (removing_edge[1], internal_node)])

            print(removing_edge)
            nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
                    with_labels=True, font_weight='bold')
            plt.show()
        else:
            internal_node = m + 1 + k
            G.add_nodes_from([(node, {"color": "red"}), (internal_node, {"color": "green"})])
            G.remove_edge(sequence[node][0], sequence[node][1])
            G.add_edges_from([(node, internal_node), (sequence[node][0], internal_node),
                              (sequence[node][1], internal_node)])
            nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
                    with_labels=True, font_weight='bold')
            plt.show()


build_pardi(sequence)
