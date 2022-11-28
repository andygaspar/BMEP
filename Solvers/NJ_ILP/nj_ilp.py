import os
import csv
import random
import sys
import warnings

import networkx as nx
import numpy as np
from Bio import Phylo
import re
import gurobipy

from matplotlib import pyplot as plt

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver

warnings.simplefilter("ignore")


class NjIlp(Solver):

    def __init__(self, d):
        super().__init__(d)
        self.path = 'Solvers/NJ_ILP/'

    def solve(self, time_limit=60):

        d_string = ''
        for i, row in enumerate(self.d):
            row_string = ['{:.19f}'.format(el) for el in row]
            line = '\n'.join(row_string)
            d_string += line + '\n'

        d_string = str(self.n_taxa) + '\n' + d_string

        with open(self.path + 'mat.mat', 'w', newline='') as csvfile:
            csvfile.write(d_string)

        os.system('julia '+self.path + 'ilp_repair_heu.jl '
                  + '--infile="Solvers/NJ_ILP/mat.mat"  --outfile="Solvers/NJ_ILP/tau.mat"  --timelimit='
                  + str(time_limit))

        self.T = np.loadtxt(self.path + "tau.mat", dtype=np.int)
        self.obj_val = self.compute_obj()



        #
        # g = nx.from_numpy_matrix(self.solution)
        # self.T = nx.floyd_warshall_numpy(g)[:self.m, :self.m].astype(int)
        # self.obj_val = self.compute_obj()

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
