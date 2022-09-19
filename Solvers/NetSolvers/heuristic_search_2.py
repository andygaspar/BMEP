import copy

import networkx as nx
import numpy as np
import torch

from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.solver import Solver


def get_idx_tuple(idx_tensor):
    return idx_tensor[0].item(), idx_tensor[1].item()


class Solution:

    def __init__(self, adj_mat, p, s):
        self.s = s
        self.adj_mat = adj_mat
        self.prob = p
        self.obj_val = None
        self.y = None


class Index:

    def __init__(self, idx_tensor, prob, sol):
        self.idx = sol.s, *get_idx_tuple(idx_tensor)
        self.idx_tensor = idx_tensor
        self.prob = prob
        self.sol = sol

    def __repr__(self):
        return self.idx

    def __hash__(self):
        return self.idx


class Distribution:

    def __init__(self, distribution_runs):
        self.idxs_dict = {}
        self.distribution_runs = distribution_runs

    def add(self, idx_tensor, sol):
        idx = sol.s, *get_idx_tuple(idx_tensor)

        if idx in list(self.idxs_dict.keys()):
            self.idxs_dict[idx].prob += sol.prob / self.distribution_runs
        else:
            self.idxs_dict[idx] = Index(idx_tensor, prob=sol.prob / self.distribution_runs, sol=sol)


class HeuristicSearch2(NetSolver):

    def __init__(self, d, net, width=2, distribution_runs=50):
        super().__init__(d, net)
        self.solution_object = None
        self.w = width
        self.distribution_runs = distribution_runs
        self.solutions = None

    def solve(self):
        adj_mat, size_mask, initial_mask, d_mask = self.initial_mats()

        self.solutions = [Solution(adj_mat, p=1, s=0)]

        with torch.no_grad():
            for i in range(3, self.n):
                distribution = Distribution(self.distribution_runs)
                for sol in self.solutions:
                    adj_mat = sol.adj_mat
                    ad_mask, mask = self.get_masks(adj_mat)
                    tau, tau_mask = self.get_tau(adj_mat, self.device)
                    for _ in range(self.distribution_runs):
                        y, _ = self.net((adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0),
                                        d_mask.unsqueeze(0),
                                        size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0),
                                        tau.unsqueeze(0),
                                        tau_mask.unsqueeze(0), None))
                        a_max = torch.argmax(y.squeeze(0))
                        idx_tensor = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'),
                                                       a_max % self.m]).to(self.device)
                        distribution.add(idx_tensor, sol=sol)
                dist_solution = sorted([idx for idx in distribution.idxs_dict.values()], key=lambda x: x.prob,
                                       reverse=True)
                dist_solution = dist_solution[:min(len(dist_solution), self.w)]
                self.solutions = []
                for s, idx in enumerate(dist_solution):
                    adj_mat = self.add_node(copy.deepcopy(idx.sol.adj_mat), idx.idx_tensor, i, self.n)
                    self.solutions.append(Solution(adj_mat, idx.prob, s))

        for sol in self.solutions:
            sol.obj_val = self.compute_obj_val(sol.adj_mat, self.d, self.n)

        solution_idx = np.argmin([sol.obj_val for sol in self.solutions])
        self.solution_object = self.solutions[solution_idx]
        self.obj_val = self.solution_object.obj_val
        self.solution = self.solutions[solution_idx].adj_mat.to("cpu").numpy()

    def check_idxs(self, idxs, step):
        for idx in idxs:
            if idx[0] >= step or idx[1] < self.n:
                idx[0] = np.random.choice(step)
                idx[1] = np.random.choice(range(self.n, self.n + step-1))
        return idxs

    def compute_obj_val(self, adj_mat, d, n):
        return self.compute_obj_val_from_adj_mat(adj_mat.to("cpu").numpy(), d.to("cpu").numpy(), n)

    def compute_idxs_distribution(self, idxs, p):
        idxs_distribution = {}

        for el in idxs:
            idx = (el[0].item(), el[1].item())

            if idx in list(idxs_distribution.keys()):
                idxs_distribution[idx] += p / self.distribution_runs
            else:
                idxs_distribution[idx] = p / self.distribution_runs
        return idxs_distribution

