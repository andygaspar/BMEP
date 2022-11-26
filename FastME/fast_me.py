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

    def __init__(self, d,
                 bme=True, nni=True, digits=None, post_processing=False, init_topology=None,
                 triangular_inequality=False, logs=False):
        super().__init__(d)
        self.path = 'FastME/fastme-2.1.6.4/'
        self.init_topology = init_topology
        self.flags = ''
        self.bme = bme
        self.nni = nni
        self.digits = digits
        self.post_processing = post_processing
        self.triangular_inequality = triangular_inequality
        self.logs = logs

    def solve(self):

        self.set_flags()

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

        if self.init_topology is not None:
            with open(self.path + 'init_topology.mat', 'w', newline='') as csvfile:
                csvfile.write(str(self.init_topology))
            os.system(self.path + "src/fastme -i " + self.path + "mat.mat " + self.flags + ' -u=init_topology.mat')
        else:
            os.system(self.path + "src/fastme -i " + self.path + "mat.mat " + self.flags)

        trees_parsed = Phylo.parse(self.path + 'mat.mat_fastme_tree.nwk', "newick")
        trees = [Phylo.to_networkx(t) for t in trees_parsed]

        tree = trees[0]
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

    def set_flags(self):

        if self.bme:
            self.flags += " -m b "
        if self.nni:
            self.flags += " -n "
        if self.post_processing:
            self.flags += " -s "
        if self.triangular_inequality:
            self.flags += " -q "
        if self.digits is not None:
            self.flags += " -f " + str(self.digits) + " "
        if not self.logs:
            self.flags += " > /dev/null"

    def change_flags(self,
                     bme=True, nni=True, digits=None, post_processing=False, triangular_inequality=False, logs=False):
        self.flags = ''
        self.bme = bme
        self.nni = nni
        self.digits = digits
        self.post_processing = post_processing
        self.triangular_inequality = triangular_inequality
        self.logs = logs
