import copy

import numpy as np
import torch.nn.init

from Solvers.solver import Solver


class NetSolver(Solver):

    def __init__(self, d, net):

        super().__init__(d)        # adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.adj_mats = []

    def solve(self):
        adj_mat, size_mask, initial_mask, d_mask = self.initial_mats()
        with torch.no_grad():
            for i in range(3, self.n):
                ad_mask, mask = self.get_masks(adj_mat)
                tau, tau_mask = self.get_tau_tensor(adj_mat, self.device)
                y, _ = self.net((adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0), d_mask.unsqueeze(0),
                                size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0),
                                tau.unsqueeze(0),
                                tau_mask.unsqueeze(0), None)
                                )
                # y, _ = self.net(adj_mat.unsqueeze(0), self.d.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0))
                a_max = torch.argmax(y.squeeze(0))
                idxs = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)
                adj_mat = self.add_node(adj_mat, idxs, new_node_idx=i, n=self.n)
                self.adj_mats.append(adj_mat.to("cpu").numpy())

        self.solution = self.adj_mats[-1].astype(int)
        self.obj_val = self.compute_obj_val_from_adj_mat(self.solution, self.d.to('cpu').numpy(), self.n)

    def initial_mats(self):
        adj_mat = self.initial_mat(self.device)
        size_mask = torch.ones_like(self.d)
        initial_mask = torch.zeros((self.m, 2)).to(self.device)
        initial_mask[:, 0] = torch.tensor([1 if i < self.n else 0 for i in range(self.m)]).to(self.device)
        initial_mask[:, 1] = torch.tensor([1 if i >= self.n else 0 for i in range(self.m)]).to(self.device)
        d_mask = copy.deepcopy(self.d)
        d_mask[d_mask > 0] = 1

        return adj_mat, size_mask, initial_mask, d_mask

    def get_masks(self, adj_mat):
        ad_mask = torch.zeros((self.d.shape[0], 2)).to(self.device)
        a = torch.sum(adj_mat, dim=-1)
        ad_mask[:, 0] = a
        ad_mask[ad_mask > 0] = 1
        ad_mask[:, 1] = torch.abs(ad_mask[:, 0] - 1)
        mask = torch.triu(adj_mat)
        return ad_mask, mask




