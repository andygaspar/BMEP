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

        self.match = None
        self.internal = [None, None]
        self.internal_list = []

    def assign(self, match):
        self.match = match

    def assign_internal(self, internal_node):
        self.internal[0] = internal_node

    def __repr__(self):
        return self.label


class Pardi:

    def __init__(self, T):
        self.T = T
        self.n = self.T.shape[0]
        self.leaves = self.make_leaves()

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

    def update_T(self, leaf, matched):
        for other_leaf in self.leaves:
            other_leaf.T_idx[matched.col] -= 1
            other_leaf.T_idx[leaf.col] = -1

    def get_pardi(self):
        print(T)
        leaves = self.leaves[::-1][:-3]
        while len(leaves) > 0:
            for leaf in leaves:
                matched_leaf = self.get_match(leaf)
                if matched_leaf is not None:
                    if matched_leaf.idx is None:
                        print(leaf, "->", matched_leaf)
                        matched_leaf.idx = leaf.insertion
                        leaf.assign(matched_leaf)
                        self.update_T(leaf, matched_leaf)

                    elif leaf.insertion < matched_leaf.idx:
                        print(leaf, "->", matched_leaf)
                        matched_leaf.idx = leaf.insertion
                        leaf.assign(matched_leaf)
                        self.update_T(leaf, matched_leaf)

                    else:
                        leaf.assign_internal(matched_leaf.idx)
                        matched_leaf.internal_list.append(leaf)
                        print("internal_node")
                    leaves.remove(leaf)


        print(self.T)


T = np.array([[0, 5, 2, 4, 5, 3],
              [5, 0, 5, 3, 2, 4],
              [2, 5, 0, 4, 5, 3],
              [4, 3, 4, 0, 3, 3],
              [5, 2, 5, 3, 0, 4],
              [3, 4, 3, 3, 4, 0]])

pardi = Pardi(T)
pardi.get_pardi()
