import copy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver


class SwaSolverNew(Solver):

    def __init__(self, d, sorted_d=False):
        super(SwaSolverNew, self).__init__(d, sorted_d)
        self.obj_val_ = None

    def solve(self, start=3, adj_mat=None):
        adj_mat = self.initial_adj_mat() if adj_mat is None else adj_mat
        adj_dict = {(i, self.n_taxa): (i, self.n_taxa) for i in range(3)}

        min_val, min_adj_dict = None, None
        for step in range(start, self.n_taxa):
            min_val = 10 ** 5

            m = (step + 1) * 2 - 2
            T = np.empty((m, m), dtype=int)
            range_m = range(m)
            range_step = range(step + 1)
            sum_idxs = [(i, j) for i in range_step for j in range_step if i < j]
            for idxs in adj_dict.values():

                new_adj_dict = self.add_node_(copy.deepcopy(adj_dict), idxs, step, self.n_taxa)
                obj_val = self.compute_obj_(new_adj_dict, step + 1, T, range_m, sum_idxs)

                if obj_val < min_val:
                    min_val, min_adj_dict = obj_val, new_adj_dict

            adj_dict = min_adj_dict

        self.solution = adj_mat
        self.obj_val = min_val


    @staticmethod
    def add_node_(adj_dict, idxs, new_node_idx, n):
        adj_dict.pop((idxs[0], idxs[1]))  # detach selected
        adj_dict[(idxs[0], n + new_node_idx - 2)] = (idxs[0], n + new_node_idx - 2)  # reattach selected to new
        adj_dict[(idxs[1], n + new_node_idx - 2)] = (idxs[1], n + new_node_idx - 2)  # reattach selected to new
        adj_dict[(new_node_idx, n + new_node_idx - 2)] = (new_node_idx, n + new_node_idx - 2)  # attach new

        return adj_dict

    def compute_obj_(self, new_adj_dict: dict, step, T, range_m, sum_idxs):
        T.fill(step)
        adj_reduced_x = [val[0] - (self.n_taxa - step) if val[0] >= self.n_taxa else val[0] for val in new_adj_dict.values()]
        adj_reduced_y = [val[1] - (self.n_taxa - step) for val in new_adj_dict.values()]
        T[adj_reduced_x, adj_reduced_y] = 1
        T[adj_reduced_y, adj_reduced_x] = 1
        np.fill_diagonal(T, 0)  # diagonal elements should be zero
        for i in range_m:
            # The second term has the same shape as A due to broadcasting
            T = np.minimum(T, T[i, :][np.newaxis, :] + T[:, i][:, np.newaxis])
        T = T[:step, :step]
        return np.sum([self.d[i, j] / self.powers[T[i, j]] for i, j in sum_idxs]) * 2





