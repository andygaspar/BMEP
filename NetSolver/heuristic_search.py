import copy

import networkx as nx
import numpy as np
import torch

from NetSolver.net_solver import NetSolver


class Solution:

    def __init__(self):
        self.adj_mat = None
        self.prob = 1
        self.obj_val = None
        self.y = None

    def compute_obj_val(self, d, n):
        g = nx.from_numpy_matrix(self.adj_mat.to("cpu").numpy())
        Tau = nx.floyd_warshall_numpy(g)[:n, :n]
        d = d.to("cpu").numpy()
        self.obj_val = np.sum([d[i, j] / 2 ** (Tau[i, j]) for i in range(n) for j in range(n)])


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
            y, _ = self.net(adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0), d_mask.unsqueeze(0),
                            size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0))
            probs, a_max = torch.topk(y, min([3, self.w]))
            probs = probs.squeeze(0)
            idxs = [torch.tensor([torch.div(a, self.m, rounding_mode='trunc'), a % self.m]).to(self.device)
                    for a in a_max.squeeze(0)]
            idxs = self.check_idxs(idxs, 3)
            for s, sol in enumerate(self.solutions[:3]):
                sol.adj_mat = self.add_node(copy.deepcopy(adj_mat), idxs[s], 3)
                sol.prob = probs[s]

            if self.w > 3:
                for i in range(3, self.w):
                    self.solutions[i].adj_mat = copy.deepcopy(self.solutions[0].adj_mat)
                    self.solutions[i].prob = copy.deepcopy(self.solutions[0].prob)

            for i in range(4, self.n):
                for sol in self.solutions:
                    adj_mat = sol.adj_mat
                    ad_mask, mask = self.get_masks(adj_mat)
                    sol.y, _ = self.net(adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0),
                                        d_mask.unsqueeze(0), size_mask.unsqueeze(0), initial_mask.unsqueeze(0),
                                        mask.unsqueeze(0))
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
                new_adj_mats = [self.add_node(copy.deepcopy(self.solutions[idxs[j][0].item()].adj_mat), idxss[j], i)
                                for j in range(self.w)]

                for j, sol in enumerate(self.solutions):
                    sol.prob = probs[j]
                    sol.adj_mat = new_adj_mats[j]

            # idxs = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)
            # adj_mat = self.add_node(adj_mat, idxs, new_node_idx=i)
            # self.adj_mats.append(adj_mat.to("cpu").numpy())
        for sol in self.solutions:
            sol.compute_obj_val(self.d, self.n)

        solution_idx = np.argmin([sol.obj_val for sol in self.solutions])
        self.solution_object = self.solutions[solution_idx]
        self.solution = self.solutions[solution_idx].adj_mat.to("cpu").numpy()

    def check_idxs(self, idxs, step):
        for idx in idxs:
            if idx[0] >= step or idx[1] < self.n:
                idx[0] = np.random.choice(step)
                idx[1] = np.random.choice(range(self.n, self.n + step-1))

        return idxs
