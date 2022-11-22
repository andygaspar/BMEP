import copy

import numpy as np
import torch

from Solvers.NetSolvers.net_solver import NetSolver
from Old.heuristic_search_2 import Solution, Distribution


class HeuristicSearch3(NetSolver):

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
            for i in range(3, self.n_taxa - 1):
                distribution = Distribution(self.distribution_runs)
                for sol in self.solutions:
                    adj_mat = sol.adj_mat
                    ad_mask, mask = self.get_masks(adj_mat)
                    tau, tau_mask = self.get_tau_tensor(adj_mat, self.device)
                    input_ = self.get_dist_input(adj_mat, ad_mask, d_mask, size_mask, initial_mask, mask, tau, tau_mask)
                    y_, _ = self.net(input_)
                    a_max_ = torch.argmax(y_, dim=-1)
                    for a in a_max_:
                        idx_tensor = torch.tensor([torch.div(a, self.m, rounding_mode='trunc'),
                                                   a % self.m]).to(self.device)
                        distribution.add(idx_tensor, sol=sol)

                dist_solution = sorted([idx for idx in distribution.idxs_dict.values()], key=lambda x: x.prob,
                                       reverse=True)
                dist_solution = dist_solution[:min(len(dist_solution), self.w)]
                self.solutions = []
                for s, idx in enumerate(dist_solution):
                    adj_mat = self.add_node(copy.deepcopy(idx.sol.adj_mat), idx.idx_tensor, i, self.n_taxa)
                    self.solutions.append(Solution(adj_mat, idx.prob, s))

        d = self.d.to("cpu").numpy()
        for sol in self.solutions:
            adj_mat = sol.adj_mat.to("cpu").numpy()
            idxs_list = np.array(np.nonzero(np.triu(adj_mat))).T
            min_val, min_adj_mat = 10**5, None
            for idxs in idxs_list:
                sol_ = self.add_node(copy.deepcopy(adj_mat), idxs, self.n_taxa - 1, self.n_taxa)
                obj_val = self.compute_obj_val_from_adj_mat(sol_, d, self.n_taxa)
                if obj_val < min_val:
                    min_val, min_adj_mat = obj_val, sol
            sol.adj_mat = min_adj_mat
            sol.obj_val = min_val

        solution_idx = np.argmin([sol.obj_val for sol in self.solutions])
        self.solution_object = self.solutions[solution_idx]
        self.obj_val = self.solution_object.obj_val
        self.solution = self.solutions[solution_idx]

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

    def get_dist_input(self, adj_mat, ad_mask, d_mask, size_mask, initial_mask, mask, tau, tau_mask):
        return (torch.cat(self.distribution_runs*[adj_mat.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[ad_mask.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[self.d.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[d_mask.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[size_mask.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[initial_mask.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[mask.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[tau.unsqueeze(0)]),
                torch.cat(self.distribution_runs*[tau_mask.unsqueeze(0)]),
                None)