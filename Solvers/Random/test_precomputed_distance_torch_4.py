import random
import time
import numpy as np
import torch

from Solvers.Random.test_precomputed_distance_torch import PrecomputeTorch
from Solvers.solver import Solver


class PrecomputeTorch3(Solver):

    def __init__(self, d, sorted_d=False, device='cuda:0'):
        super(PrecomputeTorch3, self).__init__(d, sorted_d)
        self.neighbors = None
        self.n_subtrees = 2*(2*self.n_taxa - 3)
        self.set_to_adj = None
        self.intersection = None
        self.device = device
        self.d = torch.tensor(self.d, device=self.device)
        # self.powers = torch.tensor(self.powers, device=self.device)
        self.powers = self.powers.to(self.device)
        self.subtrees_mat = None
        self.pointer_subtrees = None
        self.adj_to_set = None
        self.subtree_dist = None
        self.T_new = None

        self.mask = torch.zeros(self.n_subtrees, dtype=torch.bool, device=self.device)

    def solve(self, start=3, adj_mat=None):
        t = time.time()
        adj_mat = self.initial_adj_mat(self.device) if adj_mat is None else adj_mat
        subtrees_mat, adj_to_set  = self.initial_sub_tree_mat()
        T = self.init_tau()
        s_mat_step = 6
        # subtree_dist = self.init_sub_dist()
        for step in range(start, self.n_taxa):
            choices = 3 + (step - 3) * 2
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            rand_idxs = random.choices(range(choices))
            idxs_list = idxs_list[rand_idxs][0]
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)

            subtrees_mat, adj_to_set, T= \
                self.add_subtrees(subtrees_mat, adj_to_set, step, idxs_list, s_mat_step, T)
            s_mat_step += 4

        s = subtrees_mat[:, :self.n_taxa].to(torch.float64)


        self.subtree_dist = torch.matmul(torch.matmul(s, self.d * self.powers[T[:n, :n]]), s.T) * 2
        self.intersection = torch.matmul(s, s.T) == 0
        self.subtree_dist *= self.intersection
        self.subtrees_mat = subtrees_mat
        self.adj_to_set = adj_to_set
        self.solution = adj_mat
        self.T = T
        idxs = torch.nonzero(self.solution).T
        order = torch.argsort(self.adj_to_set[idxs[0], idxs[1]])
        self.set_to_adj = idxs.T[order]
        self.neighbors = self.compute_neighbors(self.set_to_adj, self.adj_to_set, self.solution)
        # self.device = 'cuda:0'
        print('build time', time.time() - t)
        # t = time.time()

        # print(time.time() - t)


    def initial_sub_tree_mat(self):
        subtree_mat = torch.zeros((self.n_subtrees, self.m), device=self.device, dtype=torch.long)
        adj_to_set = torch.zeros((self.m, self.m), device=self.device, dtype=torch.long)

        # 0
        subtree_mat[0, 0]  = 1
        subtree_mat[1, 1]  = 1
        subtree_mat[2, 2]  = 1

        adj_to_set[self.n_taxa, 0] = 0
        adj_to_set[self.n_taxa, 1] = 1
        adj_to_set[self.n_taxa, 2] = 2

        # 1
        subtree_mat[3, self.n_taxa] = subtree_mat[3, 1] = subtree_mat[3, 2] = 1
        subtree_mat[4, self.n_taxa] = subtree_mat[4, 0] = subtree_mat[4, 2] = 1
        subtree_mat[5, self.n_taxa] = subtree_mat[5, 0] = subtree_mat[5, 1] = 1


        adj_to_set[0, self.n_taxa] = 3
        adj_to_set[1, self.n_taxa] = 4
        adj_to_set[2, self.n_taxa] = 5

        return  subtree_mat, adj_to_set
    def add_subtrees(self, subtree_mat, adj_to_set, new_taxon_idx, idx, s_mat_step, T):

        new_internal_idx = self.n_taxa + new_taxon_idx - 2

        # singleton {k}
        subtree_mat[s_mat_step, new_taxon_idx] = 1
        adj_to_set[new_internal_idx, new_taxon_idx] = s_mat_step

        # {k} complementary
        subtree_mat[s_mat_step + 1, :new_taxon_idx] = 1
        subtree_mat[s_mat_step + 1, self.n_taxa: new_internal_idx + 1] = 1
        adj_to_set[new_taxon_idx, new_internal_idx] = s_mat_step + 1

        # update T i->j j->i distance *********************

        ij = subtree_mat[adj_to_set[idx[0], idx[1]]]
        ji = subtree_mat[adj_to_set[idx[1], idx[0]]]

        T[ij == 1] += ji
        T[ji == 1] += ij

        T[new_taxon_idx] = T[idx[0]] * ij + T[idx[1]] * ji


        # add distance k and internal k
        T[:, new_taxon_idx] = T[new_taxon_idx]
        T[new_internal_idx, :new_taxon_idx + 1] = T[:new_taxon_idx + 1, new_internal_idx] \
            = T[new_taxon_idx, :new_taxon_idx + 1] - 1
        T[new_internal_idx, self.n_taxa:new_internal_idx + 1] = T[self.n_taxa:new_internal_idx + 1, new_internal_idx] \
            = T[new_taxon_idx, self.n_taxa:new_internal_idx + 1] - 1
        T[new_internal_idx, new_internal_idx] = 0
        T[new_taxon_idx, new_internal_idx] = T[new_internal_idx, new_taxon_idx] = 1



        # i -> j to k -> j
        subtree_mat[s_mat_step + 2] = ij
        adj_to_set[new_internal_idx, idx[1]] = s_mat_step + 2


        # j -> i to k -> i
        subtree_mat[s_mat_step + 3] = ji
        adj_to_set[new_internal_idx, idx[0]] = s_mat_step + 3

        # add k and new internal to previous

        subtree_mat[:s_mat_step, new_taxon_idx] = subtree_mat[:s_mat_step, new_internal_idx] = \
            subtree_mat[:s_mat_step, idx[0]] + subtree_mat[:s_mat_step, idx[1]] - \
            (subtree_mat[:s_mat_step, idx[0]] * subtree_mat[:s_mat_step, idx[1]])


        # adjust idxs
        adj_to_set[idx[0], new_internal_idx] = adj_to_set[idx[0], idx[1]]
        adj_to_set[idx[1], new_internal_idx] = adj_to_set[idx[1], idx[0]]
        adj_to_set[idx[0], idx[1]] = adj_to_set[idx[1], idx[0]] = 0

        return subtree_mat, adj_to_set, T


    def init_sub_dist(self):
        subtree_dist = torch.zeros((self.m, self.m), device=self.device, dtype=torch.bool)
        subtree_dist[:self.n_taxa, :self.n_taxa] = self.d

        return subtree_dist

    def init_tau(self):
        T = torch.zeros((self.m, self.m), dtype=torch.long, device=self.device)
        T[0, 1] = T[0, 2] = T[1, 0] = T[1, 2] = T[2, 0] = T[2, 1] = 2
        T[0, self.n_taxa] = T[1, self.n_taxa] = T[2, self.n_taxa] =  \
            T[self.n_taxa, 0] = T[self.n_taxa, 1] = T[self.n_taxa, 2] = 1

        return T

    def compute_neighbors(self, set_to_adj, adj_to_set, adj):
        mask_j = torch.zeros_like(self.subtrees_mat)
        # get self pointing index of subtrees
        j = set_to_adj[:,1]
        i = set_to_adj[:,0]
        mask_j[range(model.n_subtrees), j] = 1

        attached_to_i = torch.nonzero(adj[i] - mask_j)

        conversion = torch.zeros_like(attached_to_i)
        k = set_to_adj[attached_to_i[:, 0], 0]
        conversion[:, 0] = k
        conversion[:, 1] = attached_to_i[:, 1]

        final_conversion = torch.stack([attached_to_i[:, 0], adj_to_set[conversion[:, 0], conversion[:, 1]]]).T

        neighbors = torch.zeros((model.n_subtrees, 2), dtype=torch.long, device=self.device)
        f = final_conversion.view((final_conversion.shape[0] // 2, 2, 2))
        neighbors[f[:, 0, 0], 0] = f[:, 0, 1]
        neighbors[f[:, 1, 0], 1] = f[:, 1, 1]

        # print(neighbors)

        return neighbors

    def check_dist(self):
        val = self.d * self.powers[self.T_new[:self.n_taxa, :self.n_taxa]]
        dist = torch.zeros_like(self.subtree_dist)
        for i in range(self.subtree_dist.shape[0]):
            for j in range(self.subtree_dist.shape[0]):
                a = self.subtrees_mat[i, :self.n_taxa].float()
                b = self.subtrees_mat[j, :self.n_taxa].float()
                idx = torch.matmul(a.unsqueeze(0).T, b.unsqueeze(0)).to(torch.long)==1
                v = val[idx].flatten().sum() * (1 - (self.subtrees_mat[i] * self.subtrees_mat[j]).sum()>0)*2
                dist[i, j] = v
        return dist

    # @torch.jit.script
    def spr(self):
        # intersections = self.intersection.clone()
        intersections = self.intersection


        # distance subtrees to taxa, useful for h computation
        sub_to_taxon_idx = torch.nonzero(self.solution[:, :self.n_taxa]).T
        sub_to_taxon = self.subtree_dist[:, self.adj_to_set[sub_to_taxon_idx[0], sub_to_taxon_idx[1]]]

        # delete complementary to each subtree as potential move
        complementary = self.set_to_adj[range(self.n_subtrees)]
        intersections[range(self.n_subtrees), self.adj_to_set[complementary[:, 1], complementary[:,0]]] = False

        # delete neighbors as potential move
        intersections[range(self.n_subtrees), self.neighbors[:, 0]] = \
            intersections[range(self.n_subtrees), self.neighbors[:, 1]] = False

        # delete from subtrees all taxa complementary (no spr moves for them)
        intersections[:, model.subtrees_mat.sum(dim=-1) == self.m - 1] = \
            intersections[model.subtrees_mat.sum(dim=-1) == self.m - 1] = False

        # neighbor 1
        inter_neighbor = intersections * intersections[self.neighbors[:, 0]]
        regrafts = torch.nonzero(inter_neighbor)
        x = self.subtrees_mat[regrafts[:, 0], : self.n_taxa]
        b = self.subtrees_mat[regrafts[:, 1], : self.n_taxa]
        a = self.subtrees_mat[self.neighbors[regrafts[:, 0], 0], : self.n_taxa]

        dist_ij = self.T[self.set_to_adj[regrafts[:, 0]][:, 0], self.set_to_adj[regrafts[:, 1]][:, 0]]

        h = 1 - x - b - a

        # LT = L(XA) + L(XB) + L(XB)
        xb = self.subtree_dist[regrafts[:, 0], regrafts[:, 1]]
        xa = self.subtree_dist[regrafts[:, 0], self.neighbors[regrafts[:, 0], 0]]
        diff_xb = xb*(1 - 1/ self.powers[dist_ij])
        diff_xa = xa * (1 - self.powers[dist_ij])

        # diff_bh = bh(1 - 2)
        diff_bh = -(sub_to_taxon[regrafts[:, 1], :] * h).sum(dim=-1)

        # diff_ah = bh(1 - 1/2)
        diff_ah = (sub_to_taxon[self.neighbors[regrafts[:, 0], 0]] * h).sum(dim=-1) / 2

        xh = sub_to_taxon[regrafts[:, 0], :] * h

        diff_xh = ((xh/(self.powers[self.T[self.set_to_adj[regrafts[:, 0]][:, 0], :self.n_taxa]])) *
                   self.powers[self.T[self.set_to_adj[regrafts[:, 1]][:, 0], :self.n_taxa]] * h).sum(dim=-1)
        diff_T = diff_xb + diff_xa + diff_bh + diff_xh + diff_ah
        min_first_side_val, min_first_side_idx = diff_T.min(0)
        min_first_side = regrafts[torch.argmin(min_first_side_idx)]

        # neighbor 2
        inter_neighbor = intersections * intersections[self.neighbors[:, 1]]
        regrafts = torch.nonzero(inter_neighbor)
        x = self.subtrees_mat[regrafts[:, 0], : self.n_taxa]
        b = self.subtrees_mat[regrafts[:, 1], : self.n_taxa]
        a = self.subtrees_mat[self.neighbors[regrafts[:, 0], 0], : self.n_taxa]

        dist_ij = self.T[self.set_to_adj[regrafts[:, 0]][:, 0], self.set_to_adj[regrafts[:, 1]][:, 0]]

        h = 1 - x - b - a

        # LT = L(XA) + L(XB) + L(XB)
        xb = self.subtree_dist[regrafts[:, 0], regrafts[:, 1]]
        xa = self.subtree_dist[regrafts[:, 0], self.neighbors[regrafts[:, 0], 0]]
        diff_xb = xb*(1 - 1/ self.powers[dist_ij])
        diff_xa = xa * (1 - self.powers[dist_ij])

        # diff_bh = bh(1 - 2)
        diff_bh = -(sub_to_taxon[regrafts[:, 1], :] * h).sum(dim=-1)

        # diff_ah = bh(1 - 1/2)
        diff_ah = (sub_to_taxon[self.neighbors[regrafts[:, 0], 0]] * h).sum(dim=-1) / 2

        xh = sub_to_taxon[regrafts[:, 0], :] * h

        diff_xh = ((xh/(self.powers[self.T[self.set_to_adj[regrafts[:, 0]][:, 0], :self.n_taxa]])) *
                   self.powers[self.T[self.set_to_adj[regrafts[:, 1]][:, 0], :self.n_taxa]] * h).sum(dim=-1)
        diff_T = diff_xb + diff_xa + diff_bh + diff_xh + diff_ah
        min_second_side_val, min_second_side_idx = diff_T.min(0)
        min_second_side = regrafts[torch.argmin(min_second_side_idx)]

        print('x')
#ll

torch.set_printoptions(precision=2, linewidth=150)

random.seed(0)
np.random.seed(0)

n = 250

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T
np.fill_diagonal(d, 0)

device = 'cpu'
# device = 'cuda:0'

model = PrecomputeTorch3(d, device=device)
t = time.time()

model.solve()
# model.plot_phylogeny()

print(model.intersection.sum(), model.intersection.shape[0]**2)

model.spr()
print(time.time() - t)


#
# s = model.subtrees_mat
# inter = model.intersection
# adj = model.solution
# set_adj = model.set_to_adj
# adj_set = model.subtrees_idx_mat
#
# mask = torch.zeros_like(s)
# # get self pointing index of subtrees
# mask[range(model.n_subtrees), set_adj[:,1]] = 1
# neigh = torch.nonzero(adj[set_adj[:,0]] - mask)
# neigh = neigh.view((neigh.shape[0]//2, 2, 2))
# n = torch.nonzero(adj[set_adj[:,0]] - mask)
# neighbors = torch.zeros((model.n_subtrees, 2), dtype=torch.long)
# neighbors[neigh[:, 0, 0], 0] = neigh[:, 0, 1]
# neighbors[neigh[:, 1, 0], 1] = neigh[:, 1, 1]
#
# j = set_adj[:,1]
# i = set_adj[:,0]
#
# mask_j = torch.zeros_like(s)
# # get self pointing index of subtrees
# mask_j[range(model.n_subtrees), set_adj[:,1]] = 1
# attached_to_i = torch.nonzero(adj[i] - mask_j)
#
# conversion = torch.zeros_like(attached_to_i)
# k = set_adj[attached_to_i[:, 0], 0]
# conversion[:, 0] = k
# conversion[:, 1] = attached_to_i[:, 1]
#
# final_conversion = torch.stack([attached_to_i[:, 0], adj_set[conversion[:, 0], conversion[:, 1]]]).T
#
# neighbors = torch.zeros((model.n_subtrees, 2), dtype=torch.long)
# f = final_conversion.view((final_conversion.shape[0]//2, 2, 2))
# neighbors[f[:, 0, 0], 0] = f[:, 0, 1]
# neighbors[f[:, 1, 0], 1] = f[:, 1, 1]
#
# f[:, 0, 0]
# f[:, 0, 1]
# neighbors