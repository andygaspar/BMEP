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

    def solve(self):
        d_string = ''
        for i, row in enumerate(self.d):
            row = map(lambda x: str(round(x, 21)), row)
            line = str(i) + ' ' + ' '.join(row)
            d_string += line + '\n'

        d_string = str(self.n_taxa) + '\n' + d_string

        with open(self.path+'mat.mat', 'w', newline='') as csvfile:
            csvfile.write(d_string)

        os.system(self.path + "src/fastme -i " + self.path + "mat.mat > /dev/null")

        with open(self.path+'mat.mat_fastme_stat.txt', 'r') as f:
            file_str = f.read()
            fastme_obj = list(map(float, re.findall('Tree length is (.+)', file_str)))
        # print(fastme_obj)

        trees_parsed = Phylo.parse(self.path+'mat.mat_fastme_tree.nwk', "newick")
        trees = [Phylo.to_networkx(t) for t in trees_parsed]

        tree = trees[0]

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
        self.T = nx.floyd_warshall_numpy(g)[:self.m, :self.m]
        self.obj_val = self.compute_obj()



