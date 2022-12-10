import copy
import time

import numpy as np
import torch

from Solvers.UCTSolver.utils.utc_utils import nni_landscape
from Solvers.solver import Solver


class cn(Solver):

    def __init__(self, d):
        super().__init__(d)

    def solve(self):
        pass
    def check_nni(self, adj_in:torch.Tensor):
        t = time.time()
        mats = nni_landscape(adj_in, self.n_taxa, self.m)
        print(time.time() - t, "time nni old")
        adj = adj_in.numpy()
        o = np.nonzero(np.triu(adj[self.n_taxa:, self.n_taxa:]))
        o = o[0] + self.n_taxa, o[1] + self.n_taxa
        k = []
        for mat in mats:
            k.append(self.compute_obj_val_from_adj_mat(mat.numpy(), self.d,self.n_taxa))

        print(min(k))

        for i in range(o[0].shape[0]):
            a, b = o[0][i], o[1][i]
            to_swap_a = np.nonzero(adj[a])[0]
            to_swap_a = np.array([el for el in to_swap_a if el != b])
            to_swap_b = np.nonzero(adj[b])[0]
            to_swap_b = np.array([el for el in to_swap_b if el != a])

            ad = copy.deepcopy(adj)
            ad[to_swap_a[0], a] = ad[a, to_swap_a[0]] = ad[to_swap_b[0], b] = ad[b, to_swap_b[0]] = 0
            ad[to_swap_a[0], b] = ad[b, to_swap_a[0]] = ad[to_swap_b[0], a] = ad[a, to_swap_b[0]] = 1

            objval = self.compute_obj_val_from_adj_mat(ad, self.d, self.n_taxa)
            # print(objval)

            ad = copy.deepcopy(adj)
            ad[to_swap_a[1], a] = ad[a, to_swap_a[1]] = ad[to_swap_b[0], b] = ad[b, to_swap_b[0]] = 0
            ad[to_swap_a[1], b] = ad[b, to_swap_a[1]] = ad[to_swap_b[0], a] = ad[a, to_swap_b[0]] = 1

            objval = self.compute_obj_val_from_adj_mat(ad, self.d, self.n_taxa)
            # print(objval)

        t = time.time()
        idxs = torch.nonzero(torch.triu(adj_in[self.n_taxa:, self.n_taxa:]))
        idxs += self.n_taxa
        adj_mats = adj_in.unsqueeze(0).repeat(idxs.shape[0] * 2, 1, 1)

        half_one = range(idxs.shape[0])
        half_two = range(idxs.shape[0], idxs.shape[0] * 2)
        evens, odds = range(0, idxs.shape[0] * 2, 2), range(1, idxs.shape[0] * 2 + 1, 2)

        adj_mats[half_one, idxs[:, 0], idxs[:, 1]] = adj_mats[half_one, idxs[:, 1], idxs[:, 0]] = 0 # hide temporally to get subtrees


        to_swap_a = torch.nonzero(adj_mats[half_one, idxs[:, 0], :])
        to_swap_b = torch.nonzero(adj_mats[half_one, idxs[:, 1], :])

        # fix hiding and get back correct matrices
        adj_mats[half_one, idxs[:, 0], idxs[:, 1]] = adj_mats[half_one, idxs[:, 1], idxs[:, 0]] = 1


        adj_mats[half_one, to_swap_a[evens, 1], idxs[:, 0]] = \
            adj_mats[half_one, idxs[:, 0], to_swap_a[evens, 1]] = 0
        adj_mats[half_one, to_swap_b[evens, 1], idxs[:, 1]] = \
            adj_mats[half_one, idxs[:, 1], to_swap_b[evens, 1]] = 0

        adj_mats[half_one, to_swap_a[evens, 1], idxs[:, 1]] = \
            adj_mats[half_one, idxs[:, 1], to_swap_a[evens, 1]] = 1
        adj_mats[half_one, to_swap_b[evens, 1], idxs[:, 0]] = \
            adj_mats[half_one, idxs[:, 0], to_swap_b[evens, 1]] = 1


        adj_mats[half_two, to_swap_a[odds, 1], idxs[:, 0]] = \
            adj_mats[half_two, idxs[:, 0], to_swap_a[odds, 1]] = 0
        adj_mats[half_two, to_swap_b[evens, 1], idxs[:, 1]] = \
            adj_mats[half_two, idxs[:, 1], to_swap_b[evens, 1]] = 0

        adj_mats[half_two, to_swap_a[odds, 1], idxs[:, 1]] = \
            adj_mats[half_two, idxs[:, 1], to_swap_a[odds, 1]] = 1
        adj_mats[half_two, to_swap_b[evens, 1], idxs[:, 0]] = \
            adj_mats[half_two, idxs[:, 0], to_swap_b[evens, 1]] = 1

        print(time.time() - t, "time nni")

        Tau = torch.full_like(adj_mats, self.n_taxa)
        Tau[adj_mats > 0] = 1
        diag = torch.eye(adj_mats.shape[1]).repeat(adj_mats.shape[0], 1, 1).bool()
        Tau[diag] = 0  # diagonal elements should be zero
        for i in range(adj_mats.shape[1]):
            # The second term has the same shape as Tau due to broadcasting
            Tau = torch.minimum(Tau, Tau[:, i, :].unsqueeze(1).repeat(1, adj_mats.shape[1], 1)
                                + Tau[:, :, i].unsqueeze(2).repeat(1, 1, adj_mats.shape[1]))

        d = torch.tensor(self.d)
        results =  (d * 2 ** (-Tau[:, :self.n_taxa, :self.n_taxa])).reshape(adj_mats.shape[0], -1).sum(dim=-1)
        print(torch.min(results).item())
        # for r in results:
        #     print(r.item())

