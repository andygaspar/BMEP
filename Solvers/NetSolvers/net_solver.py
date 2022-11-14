import copy

import numpy as np
import torch.nn.init

from Solvers.solver import Solver


class NetSolver(Solver):

    def __init__(self, d, net):
        super().__init__(d)
        d_with_internals = torch.zeros((self.m, self.m))
        d_with_internals[:self.n_taxa, :self.n_taxa] = d if type(d) == torch.Tensor else torch.tensor(d)
        self.d = d_with_internals.to(torch.float).to(self.device)
        self.net = net
        self.adj_mats = []

    def solve(self):
        adj_mat = self.initial_adj_mat(self.device)
        with torch.no_grad():
            for i in range(3, self.n_taxa):
                tau = self.get_tau(adj_mat.to("cpu").numpy(), self.device)
                net_input = self.net.get_net_input(adj_mat, self.d, tau, self.m, self.n_taxa, i)
                y, _ = self.net(net_input)
                a_max = torch.argmax(y.squeeze(0))
                idxs = self.get_idx_from_prob(a_max)
                adj_mat = self.add_node(adj_mat, idxs, new_node_idx=i, n=self.n_taxa)
                self.adj_mats.append(adj_mat.to("cpu").numpy())

        self.solution = self.adj_mats[-1].astype(int)
        self.obj_val = self.compute_obj_val_from_adj_mat(self.solution, self.d.to('cpu').numpy(), self.n_taxa)

    def get_idx_from_prob(self, a_max):
        return torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)
