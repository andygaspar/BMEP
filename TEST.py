import numpy as np
import string

import copy
import string

import networkx as nx
from matplotlib import pyplot as plt

from alg_ import get_pardi
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

T = np.array([[0, 4, 2, 5, 5, 4, 4],
              [4, 0, 4, 5, 5, 2, 4],
              [2, 4, 0, 5, 5, 4, 4],
              [5, 5, 5, 0, 2, 5, 3],
              [5, 5, 5, 2, 0, 5, 3],
              [4, 2, 4, 5, 5, 0, 4],
              [4, 4, 4, 3, 3, 4, 0]])


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
    for k, node in enumerate(sequence):
        if len(node[1]) < 2:
            internal_node = m + 1 + k
            G.add_nodes_from([(node[0], {"color": "red"}), (internal_node, {"color": "green"})])
            removing_edge = list(G.edges(node[1][0]))[0]
            G.remove_edge(removing_edge[0], removing_edge[1])
            G.add_edges_from([(node[0], internal_node), (node[1][0], internal_node), (removing_edge[1], internal_node)])

            print(removing_edge)
            nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes],
                    with_labels=True, font_weight='bold')
            plt.show()
        else:
            internal_node = m + 1 + k
            G.add_nodes_from([(node[0], {"color": "red"}), (internal_node, {"color": "green"})])
            G.remove_edge(node[1][0], node[1][1])
            G.add_edges_from([(node[0], internal_node), (node[1][0], internal_node), (node[1][2], internal_node)])


build_pardi(sequence)