import copy
import string

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from Pardi.Leaf import Leaf


class Pardi:

    def __init__(self, T):
        self.T = T
        self.n = self.T.shape[0]
        self.leaves = self.make_leaves()
        self.compute_pardi()

        self.graph = None

    def make_leaves(self):
        leaves = []
        labels = [s for s in string.ascii_uppercase[:self.n]]
        for i in range(3):
            leaves.append(Leaf(labels[i], i, 1, self.T[i]))
        for i in range(3, self.n):
            leaves.append(Leaf(labels[i], i, i - 1, self.T[i]))
        return leaves

    def get_leaf(self, col):
        return self.leaves[col]

    def get_match(self, leaf):
        match = np.where(leaf.T_idx == 2)[0]
        return self.get_leaf(match[0]) if match.shape[0] > 0 else None

    def update_distance(self, leaf, matched):
        self.T[matched.col] -= 1
        self.T[:, matched.col] -= 1
        self.T[leaf.col] = -1
        self.T[:, leaf.col] = -1

    def compute_pardi(self):
        leaves = self.leaves[::-1][:-3]
        while leaves:
            for leaf in leaves:
                matched_leaf = self.get_match(leaf)
                if matched_leaf is not None:
                    if matched_leaf.idx is None or leaf.insertion < matched_leaf.idx:
                        matched_leaf.idx = leaf.insertion
                        leaf.assign(matched_leaf)
                        self.update_distance(leaf, matched_leaf)

                    else:
                        matched_leaf.internal_list.append(leaf)
                        self.update_distance(leaf, matched_leaf)
                    leaves.remove(leaf)
                    break

        for leaf in self.leaves[:3]:
            if leaf.internal_list:
                leaf.assign_internals()

        for leaf in self.leaves:
            leaf.node = leaf.node if leaf.node is not None else leaf.insertion

    def get_pardi(self, print_=False):
        for leaf in self.leaves[3:]:
            leaf.get_assignment(print_)

    def get_graph(self, show=False):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([("A", {"color": "red"}), ("B", {"color": "red"}),
                                   ("C", {"color": "red"}), (1, {"color": "green"})])
        self.graph.add_edges_from([("A", 1), ("B", 1), ("C", 1)])

        for k, leaf in enumerate(self.leaves[3:]):
            assignment = leaf.get_assignment()
            if not type(assignment) == Leaf:
                internal_node = k + 2
                self.graph.add_nodes_from([(leaf.label, {"color": "red"}), (internal_node, {"color": "green"})])
                self.graph.remove_edge(assignment[0], assignment[1])
                self.graph.add_edges_from([(leaf.label, internal_node), (assignment[0], internal_node),
                                           (assignment[1], internal_node)])

            else:
                internal_node = k + 2
                self.graph.add_nodes_from([(leaf.label, {"color": "red"}), (internal_node, {"color": "green"})])
                removing_edge = list(self.graph.edges(assignment.label))[0]
                self.graph.remove_edge(removing_edge[0], removing_edge[1])
                self.graph.add_edges_from([(leaf.label, internal_node), (assignment.label, internal_node),
                                           (internal_node, removing_edge[1])])
        if show:
            nx.draw(self.graph, node_color=[self.graph.nodes[node]["color"] for node in self.graph.nodes],
                    with_labels=True, font_weight='bold')
            plt.show()

        return self.graph






