import copy
import numpy as np
import torch

from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.solver import Solver


class Solution:

    def __init__(self):
        self.adj_mat = None
        self.prob = 1
        self.obj_val = None
        self.y = None


class HeuristicSearch(NetSolver):

    def __init__(self, d, net, width=2):
        super().__init__(d, net)
        self.solution_object = None
        self.w = width
        self.solutions = [Solution() for _ in range(self.w)]

    def solve(self):
        adj_mat, size_mask, initial_mask, d_mask = self.initial_mats()
        with torch.no_grad():
            ad_mask, mask = self.get_masks(adj_mat)
            tau, tau_mask = self.get_tau_tensor(adj_mat, self.device)
            y, _ = self.net((adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0),
                             d_mask.unsqueeze(0),
                             size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0),
                             tau.unsqueeze(0),
                             tau_mask.unsqueeze(0), None))
            probs, a_max = torch.topk(y, min([3, self.w]))
            probs = probs.squeeze(0)
            idxs = [torch.tensor([torch.div(a, self.m, rounding_mode='trunc'), a % self.m]).to(self.device)
                    for a in a_max.squeeze(0)]
            idxs = self.check_idxs(idxs, 3)
            for s, sol in enumerate(self.solutions[:3]):
                sol.adj_mat = self.add_node(copy.deepcopy(adj_mat), idxs[s], 3, self.n)
                sol.prob = probs[s]

            if self.w > 3:
                for i in range(3, self.w):
                    self.solutions[i].adj_mat = copy.deepcopy(self.solutions[0].adj_mat)
                    self.solutions[i].prob = copy.deepcopy(self.solutions[0].prob)

            for i in range(4, self.n):
                for sol in self.solutions:
                    adj_mat = sol.adj_mat
                    ad_mask, mask = self.get_masks(adj_mat)
                    tau, tau_mask = self.get_tau_tensor(adj_mat, self.device)
                    sol.y, _ = self.net((adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0),
                                     d_mask.unsqueeze(0),
                                     size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0),
                                     tau.unsqueeze(0),
                                     tau_mask.unsqueeze(0), None))
                    sol.y *= sol.prob
                p = torch.cat([sol.y for sol in self.solutions])
                # print(p)
                probs, a_max = torch.topk(p.flatten(), self.w)
                # probs = probs.squeeze(0)
                idxs = [torch.tensor([torch.div(a, self.m ** 2, rounding_mode='trunc'),
                                      torch.div(a % self.m ** 2, self.m, rounding_mode='trunc'),
                                      a % self.m]).to(self.device)
                        for a in a_max]
                idxss = self.check_idxs([idx[1:] for idx in idxs], i)
                new_adj_mats = [self.add_node(copy.deepcopy(self.solutions[idxs[j][0].item()].adj_mat),
                                              idxss[j], i, self.n)
                                for j in range(self.w)]

                for j, sol in enumerate(self.solutions):
                    sol.prob = probs[j]
                    sol.adj_mat = new_adj_mats[j]

            # idxs = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)
            # adj_mat = self.add_node(adj_mat, idxs, new_node_idx=i)
            # self.adj_mats.append(adj_mat.to("cpu").numpy())
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

