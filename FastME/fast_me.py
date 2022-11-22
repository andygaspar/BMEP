import os
import csv
import random
import sys
import warnings

import networkx as nx
import numpy as np
from Bio import Phylo
import re

from matplotlib import pyplot as plt

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver

warnings.simplefilter("ignore")


class FastMeSolver(Solver):

    def __init__(self, d):
        super().__init__(d)
        self.path = 'FastME/fastme-2.1.6.4/'
        self.flags = ''

    def solve(self, bme=True, nni=True, digits=None, triangular_inequality=False, logs=False):

        self.set_flags(bme, nni, digits, triangular_inequality, logs)

        # mettere tutte flag bene e controllare taxaaddbal
        d_string = ''
        for i, row in enumerate(self.d):
            # row_ = map(lambda x: str(round(x, 21)), row)
            # line = str(i) + ' ' + ' '.join(row_)
            row_string = ['{:.19f}'.format(el) for el in row]
            line = str(i) + ' ' + ' '.join(row_string)
            d_string += line + '\n'

        d_string = str(self.n_taxa) + '\n' + d_string

        with open(self.path + 'mat.mat', 'w', newline='') as csvfile:
            csvfile.write(d_string)

        os.system(self.path + "src/fastme -i " + self.path + "mat.mat " + self.flags)

        trees_parsed = Phylo.parse(self.path + 'mat.mat_fastme_tree.nwk', "newick")
        trees = [Phylo.to_networkx(t) for t in trees_parsed]

        tree = trees[0]

        #os.system(self.path + "src/fastme -help")

        # nx.draw(tree, with_labels=True)
        # plt.show()
        tree.edges()
        adj_mat = nx.adjacency_matrix(tree).toarray()
        adj_mat[adj_mat != 0] = 1
        adj_mat = adj_mat.astype(int)

        nodes = []
        for n in tree.nodes:
            nodes.append(str(n))
        idx = np.argsort(nodes)

        self.solution = adj_mat[idx][:, idx]
        g = nx.from_numpy_matrix(self.solution)
        self.T = nx.floyd_warshall_numpy(g)[:self.m, :self.m].astype(int)
        self.obj_val = self.compute_obj()

    def set_flags(self, bme, nni, digits, triangular_inequality, logs):

        if bme:
            self.flags += " -m b "
        if nni:
            self.flags += " -n "
        if triangular_inequality:
            self.flags += " -q "
        if digits is not None:
            self.flags += " -f " + str(digits) + " "
        if not logs:
            self.flags += " > /dev/null"
