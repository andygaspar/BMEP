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


class HeuristicSearchDropOut(NetSolver):

    def __init__(self, d, net, width=2, distribution_runs=50):
        super().__init__(d, net)
        self.solution_object = None
        self.w = width
        self.distribution_runs = distribution_runs
        self.solutions = None

    def solve(self):
        adj_mat = self.initial_adj_mat(self.device)

        self.solutions = [Solution(adj_mat, p=1, s=0)]

        with torch.no_grad():
            for i in range(3, self.n_taxa):
                distribution = Distribution(self.distribution_runs)
                for sol in self.solutions:
                    adj_mat = sol.adj_mat
                    tau = self.get_tau(adj_mat.to("cpu").numpy(), device=self.device)
                    net_input = self.net.get_net_input(adj_mat, self.d, tau, self.m, self.n_taxa, step=i)
                    input_ = self.get_dist_input(net_input)
                    y_, _ = self.net(input_)
                    a_max_ = torch.argmax(y_, dim=-1)
                    for a in a_max_:
                        idx_tensor = torch.tensor([torch.div(a, self.m, rounding_mode='trunc'),
                                                   a % self.m]).to(self.device)
                        distribution.add(idx_tensor, sol=sol)

                dist_solution = sorted([idx for idx in distribution.idxs_dict.values()], key=lambda x: x.prob,
                                       reverse=True)
                # l = len(dist_solution)
                dist_solution = dist_solution[:min(len(dist_solution), self.w)]
                # print(len(dist_solution), [val.prob for val in dist_solution])
                self.solutions = []
                for s, idx in enumerate(dist_solution):
                    adj_mat = self.add_node(copy.deepcopy(idx.sol.adj_mat), idx.idx_tensor, i, self.n_taxa)
                    self.solutions.append(Solution(adj_mat, idx.prob, s))

        for sol in self.solutions:
            sol.obj_val = self.compute_obj_val(sol.adj_mat, self.d, self.n_taxa)

        solution_idx = np.argmin([sol.obj_val for sol in self.solutions])
        self.solution_object = self.solutions[solution_idx]
        self.obj_val = self.solution_object.obj_val
        self.solution = self.solutions[solution_idx].adj_mat.to("cpu").numpy()

    def check_idxs(self, idxs, step):
        for idx in idxs:
            if idx[0] >= step or idx[1] < self.n_taxa:
                idx[0] = np.random.choice(step)
                idx[1] = np.random.choice(range(self.n_taxa, self.n_taxa + step - 1))
        return idxs

    def compute_obj_val(self, adj_mat, d, n_taxa):
        return self.compute_obj_val_from_adj_mat(adj_mat.to("cpu").numpy(), d.to("cpu").numpy(), n_taxa)

    def compute_idxs_distribution(self, idxs, p):
        idxs_distribution = {}

        for el in idxs:
            idx = (el[0].item(), el[1].item())

            if idx in list(idxs_distribution.keys()):
                idxs_distribution[idx] += p / self.distribution_runs
            else:
                idxs_distribution[idx] = p / self.distribution_runs
        return idxs_distribution

    def get_dist_input(self, net_input):
        return tuple([torch.cat(self.distribution_runs*[net_input[i]]) for i in range(len(net_input))])
