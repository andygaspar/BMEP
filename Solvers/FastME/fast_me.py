import os
import warnings

import networkx as nx
import numpy as np

from matplotlib import pyplot as plt

from Solvers.solver import Solver
from Solvers.FastME.pharser_newik.newwik_handler import get_adj_from_nwk, compute_newick

warnings.simplefilter("ignore")


class FastMeSolver(Solver):

    def __init__(self, d,
                 bme=True, nni=True, digits=None, post_processing=False, init_topology=None,
                 triangular_inequality=False, logs=False):
        super().__init__(d)
        self.path = 'Solvers/FastME/fastme-2.1.6.4/'
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
            row_string = ['{:.19f}'.format(el) for el in row]
            line = str(i) + ' ' + ' '.join(row_string)
            d_string += line + '\n'

        d_string = str(self.n_taxa) + '\n' + d_string

        with open(self.path + 'mat.mat', 'w', newline='') as csvfile:
            csvfile.write(d_string)

        if self.init_topology is not None:
            with open(self.path + 'init_topology.nwk', 'w', newline='') as csvfile:
                csvfile.write(compute_newick(self.init_topology))

        os.system(self.path + "src/fastme -i " + self.path + "mat.mat " + self.flags)

        adj_mat = get_adj_from_nwk(self.path + 'mat.mat_fastme_tree.nwk')
        # self.check_mat(adj_mat)
        self.solution = adj_mat
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
        if self.init_topology is not None:
            self.flags += ' -u ' + self.path + 'init_topology.nwk'
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

    def check_mat(self, adj_mat):
        taxa = np.array_equal(adj_mat[:self.n_taxa, self.n_taxa:].sum(axis=1), np.ones(self.n_taxa))
        internals = np.array_equal(adj_mat[self.n_taxa:, :].sum(axis=1), np.ones(self.n_taxa)*3)
        graph = nx.from_numpy_matrix(adj_mat)
        pos = nx.spring_layout(graph)

        nx.draw(graph, pos=pos, node_color=['green' if i < self.n_taxa else 'red' for i in range(self.m)],
                with_labels=True, font_weight='bold')
        plt.show()
        return taxa*internals



