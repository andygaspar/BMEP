import copy
import string

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np


class Leaf:

    def __init__(self, label, col, insertion, T_idx):
        self.label = label
        self.col = col
        self.T_idx = T_idx
        self.insertion = insertion
        self.idx = None

        self.matched = False
        self.match = None
        self.internal = [None, None]
        self.internal_list = []

        self.node = None

    def get_assignment(self, print_=False):
        assignment = self.match if self.match is not None else self.internal
        if print_:
            print(self.label, "->", assignment)
        return assignment

    def assign_internals(self):
        sorted_internal = sorted(self.internal_list, reverse=True)
        for leaf in sorted_internal:
            for i in range(len(sorted_internal)):
                if leaf == self.internal_list[i]:
                    if i == 0:
                        leaf.internal[0] = self.idx
                    else:
                        leaf.internal[0] = self.internal_list[i - 1].insertion
                    if i < len(self.internal_list) - 1:
                        leaf.internal[1] = self.internal_list[i + 1].insertion
                    else:
                        leaf.internal[1] = self.insertion

                    self.internal_list.remove(leaf)
                    break

    def assign(self, match):
        self.match = match
        if not match.matched:
            match.node = self.insertion
            match.matched = True
        if self.internal_list:
            self.assign_internals()

    def __repr__(self):
        return self.label

    def __lt__(self, other):
        return self.col < other.col

    def __eq__(self, other):
        return self.col == other.col


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
            leaves.append(Leaf(labels[i], i, 1, T[i]))
        for i in range(3, self.n):
            leaves.append(Leaf(labels[i], i, i - 1, T[i]))
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
        nx.draw(self.graph, node_color=[self.graph.nodes[node]["color"] for node in self.graph.nodes],
                with_labels=True, font_weight='bold')
        plt.show()
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


T = np.array([[0, 5, 2, 4, 5, 3],
              [5, 0, 5, 3, 2, 4],
              [2, 5, 0, 4, 5, 3],
              [4, 3, 4, 0, 3, 3],
              [5, 2, 5, 3, 0, 4],
              [3, 4, 3, 3, 4, 0]])

T = np.array([[0, 3, 8, 8, 8, 3, 5, 3, 6, 4, 8],
              [3, 0, 9, 9, 9, 2, 6, 4, 7, 5, 9],
              [8, 9, 0, 4, 4, 9, 5, 7, 4, 6, 2],
              [8, 9, 4, 0, 2, 9, 5, 7, 4, 6, 4],
              [8, 9, 4, 2, 0, 9, 5, 7, 4, 6, 4],
              [3, 2, 9, 9, 9, 0, 6, 4, 7, 5, 9],
              [5, 6, 5, 5, 5, 6, 0, 4, 3, 3, 5],
              [3, 4, 7, 7, 7, 4, 4, 0, 5, 3, 7],
              [6, 7, 4, 4, 4, 7, 3, 5, 0, 4, 4],
              [4, 5, 6, 6, 6, 5, 3, 3, 4, 0, 6],
              [8, 9, 2, 4, 4, 9, 5, 7, 4, 6, 0]])

# T = np.array([[0, 3, 9, 8, 8, 9, 4, 6, 5, 8, 3, 3],
#               [3, 0,10, 9, 9,10, 5, 7, 6, 9, 2, 4],
#               [9,10, 0, 5, 5, 2, 7, 5, 6, 3,10, 8],
#               [8, 9, 5, 0, 2, 5, 6, 4, 5, 4, 9, 7],
#               [8, 9, 5, 2, 0, 5, 6, 4, 5, 4, 9, 7],
#               [9,10, 2, 5, 5, 0, 7, 5, 6, 3,10, 8],
#               [4, 5, 7, 6, 6, 7, 0, 4, 3, 6, 5, 3],
#               [6, 7, 5, 4, 4, 5, 4, 0, 3, 4, 7, 5],
#               [5, 6, 6, 5, 5, 6, 3, 3, 0, 5, 6, 4],
#               [8, 9, 3, 4, 4, 3, 6, 4, 5, 0, 9, 7],
#               [3, 2,10, 9, 9,10, 5, 7, 6, 9, 0, 4],
#               [3, 4, 8, 7, 7, 8, 3, 5, 4, 7, 4, 0]])


pardi = Pardi(T)
pardi.get_pardi(True)
# pardi.get_graph(plot=True)
p = 0
pardi.build_pardi()
