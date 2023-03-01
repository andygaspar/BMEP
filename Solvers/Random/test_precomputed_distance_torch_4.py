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
        self.powers = torch.tensor(self.powers, device=self.device)
        # self.powers = self.powers.to(self.device)
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
        self.obj_val = (self.powers[self.T[:self.n_taxa, :self.n_taxa]]*self.d).sum().item()
        print('obj val', self.obj_val)
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

    def spr(self):
        # intersections = self.intersection.clone()
        intersections = self.intersection


        # distance subtrees to taxa, useful for h computation
        sub_to_taxon_idx = torch.nonzero(self.solution[:, :self.n_taxa]).T
        _, order = torch.sort(sub_to_taxon_idx[1])
        sub_to_taxon_idx = self.adj_to_set[sub_to_taxon_idx[0], sub_to_taxon_idx[1]][order]
        sub_to_taxon = self.subtree_dist[:, sub_to_taxon_idx]

        # delete complementary to each subtree as potential move
        complementary = self.set_to_adj[range(self.n_subtrees)]
        intersections[range(self.n_subtrees), self.adj_to_set[complementary[:, 1], complementary[:,0]]] = False

        del(complementary)

        # delete neighbors as potential move
        intersections[range(self.n_subtrees), self.neighbors[:, 0]] = \
            intersections[range(self.n_subtrees), self.neighbors[:, 1]] = False

        # delete from subtrees all taxa complementary (no spr moves for them)
        intersections[:, model.subtrees_mat.sum(dim=-1) == self.m - 1] = \
            intersections[model.subtrees_mat.sum(dim=-1) == self.m - 1] = False



        # neighbor 1

        max_first_side_val, max_first_side_move, diff_T = self.spr_side(intersections, sub_to_taxon, a_side=0)

        # inter_neighbor = intersections * intersections[self.neighbors[:, 0]]
        # regrafts = torch.nonzero(inter_neighbor)
        #
        # diff = []
        # diff_new = []
        # d_xh = []
        # for i in range(regrafts.shape[0]):
        #     d_result, dxh = self.test_diff(regrafts[i], a_side_idx=0)
        #     diff_new.append(d_result)
        #     d_xh.append(dxh)
        #     new_adj = self.move(regrafts[i], a_side_idx=0, adj_mat=self.solution.clone())
        #     T = self.get_tau(new_adj)
        #     diff.append(self.obj_val - (self.powers[torch.tensor(T)]*self.d).sum().item())
        # d_xh = torch.tensor(d_xh)
        #
        # for i in range(regrafts.shape[0]):
        #     print(diff_T[i].item(), diff[i], regrafts[i], diff_new[i])

        # neightbor 2

        max_second_side_val, max_second_side_move, diff_T = self.spr_side(intersections, sub_to_taxon, a_side=1)

        # inter_neighbor = intersections * intersections[self.neighbors[:, 1]]
        # regrafts = torch.nonzero(inter_neighbor)
        # diff = []
        # diff_new = []
        # d_xh = []
        # for i in range(regrafts.shape[0]):
        #     d_result, dxh = self.test_diff(regrafts[i], a_side_idx=1)
        #     diff_new.append(d_result)
        #     d_xh.append(dxh)
        #     new_adj = self.move(regrafts[i], a_side_idx=1, adj_mat=self.solution.clone())
        #     T = self.get_tau(new_adj)
        #     diff.append(self.obj_val - (self.powers[torch.tensor(T)]*self.d).sum().item())
        # d_xh = torch.tensor(d_xh)
        #
        # # neightbor 2
        # print("side 2")
        # for i in range(regrafts.shape[0]):
        #     print(diff_T[i].item(), diff[i], regrafts[i], diff_new[i])


        max_side = torch.argmax(torch.stack([max_first_side_val, max_second_side_val], dim=-1)) # a side idx
        best_move = torch.cat([max_first_side_move.unsqueeze(0), max_second_side_move.unsqueeze(0)])

        return best_move[max_side], max_side


    def spr_side(self, intersections, sub_to_taxon, a_side):

        inter_neighbor = intersections * intersections[self.neighbors[:, a_side]]
        regrafts = torch.nonzero(inter_neighbor)

        x = regrafts[:, 0]
        b = regrafts[:, 1]
        a = self.neighbors[x, a_side]

        b_adj = self.set_to_adj[b]
        b_c = self.adj_to_set[b_adj[:, 1], b_adj[:, 0]]

        a_adj = self.set_to_adj[a]
        a_c = self.adj_to_set[a_adj[:, 1], a_adj[:, 0]]

        dist_ij = self.T[self.set_to_adj[x][:, 0], self.set_to_adj[b][:, 0]]

        h = 1 - self.subtrees_mat[x] - self.subtrees_mat[b] - self.subtrees_mat[a]
        h = h[:, :self.n_taxa]

        # LT = L(XA) + L(XB) + L(XB)
        xb = self.subtree_dist[x, b]
        xa = self.subtree_dist[x, a]
        diff_xb = xb * (1 - 1/ self.powers[dist_ij])
        diff_xa = xa * (1 - self.powers[dist_ij])

        # diff_bh = bh(1 - 1/2)
        diff_bh = (self.subtree_dist[b, b_c] - xb - self.subtree_dist[a, b]) / 2

        # diff_ah = ah(1 - 2)
        diff_ah = -(self.subtree_dist[a, a_c] - self.subtree_dist[a, b] - xa)


        p1 = self.powers[self.T[self.set_to_adj[x][:, 0], :self.n_taxa]]
        p2 = self.powers[self.T[self.set_to_adj[b][:, 1], :self.n_taxa]]

        diff_xh = (sub_to_taxon[x, :] * ( (p1 - p2) / p1) * h).sum(dim=-1)

        diff_T = diff_xb + diff_xa + diff_bh + diff_xh + diff_ah
        max_side_val, max_side_idx = diff_T.max(0)

        max_side_move = regrafts[max_side_idx]

        return max_side_val, max_side_move , diff_T

    def test_diff(self, selected_move, a_side_idx):
        x = selected_move[0]
        x_adj = self.set_to_adj[x]
        x_c = self.adj_to_set[x_adj[1], x_adj[0]]

        b = selected_move[1]
        b_adj = self.set_to_adj[b]
        b_c = self.adj_to_set[b_adj[1], b_adj[0]]

        a = self.neighbors[x, a_side_idx]
        a_adj = self.set_to_adj[a]
        a_c = self.adj_to_set[a_adj[1], a_adj[0]]

        ab = self.subtree_dist[a, b]


        #test
        h =  1 - self.subtrees_mat[x] - self.subtrees_mat[b] - self.subtrees_mat[a]
        h = h[:self.n_taxa]

        hx = self.h_dist(x, h)
        ah = self.h_dist(a, h)
        hb = self.h_dist(b, h)

        # LT = L(XA) + L(XB) + L(XB)
        xb = self.subtree_dist[x, b] #ok
        dist_xb = self.subset_dist(x, b)

        xa = self.subtree_dist[x, a] #ok
        dist_xa = self.subset_dist(x, b)

        dist_ij = self.T[self.set_to_adj[x, 0], self.set_to_adj[b, 0]]
        diff_xb = xb * (1 - 1/self.powers[dist_ij])
        diff_xa = xa * (1 - self.powers[dist_ij])

        # diff_bh = bh(1 - 1/2),  bh = b<->b_c - xb - ba
        diff_bh = (self.subtree_dist[b, b_c] - xb - self.subtree_dist[a, b])/2
        bh = self.subtree_dist[b, b_c] - xb - self.subtree_dist[a, b]
        k = bh.item()
        kk = hb.item()

        # diff_ah = ah(1 - 2), ah = a<->a_c - ab - xa
        diff_ah = -(self.subtree_dist[a, a_c] - self.subtree_dist[a, b] - xa)


        # xh = x<->x_c - xa - xb
        xh = self.subtree_dist[x, x_c] - xa - xb
        x_set_idx = torch.nonzero(self.subtrees_mat[x, :self.n_taxa]).flatten()

        l = self.subset_len(a) + self.subset_len(b) + self.subset_len(x) + self.h_len(h) \
            + ab + xa + ah + xh + xb + hb
        l = l.item()

        diff_xh = xh - (self.powers[self.T[b_adj[1], :self.n_taxa]] * self.d[x_set_idx] * h).sum()
        diff_T = (diff_xb + diff_xa + diff_bh + diff_xh + diff_ah).item()
        # -0.06689979168715987
        return diff_T, diff_xh.item()

    def subset_dist(self, set_v, set_w):
        vw = torch.matmul(self.subtrees_mat[set_v, :self.n_taxa].to(torch.float64),
                            self.powers[self.T[:self.n_taxa, :self.n_taxa]] * self.d)
        return torch.matmul(vw, self.subtrees_mat[set_w, :self.n_taxa].to(torch.float64)) * 2

    def h_dist(self, subtree, h):
        vw = torch.matmul(self.subtrees_mat[subtree, :self.n_taxa].to(torch.float64),
                          self.powers[self.T[:self.n_taxa, :self.n_taxa]] * self.d)
        return torch.matmul(vw, h.to(torch.float64)) * 2

    def subset_len(self, s):
        return self.subset_dist(s, s)/2

    def tree_len(self):
        full_set = torch.ones(self.n_taxa, dtype=torch.float64)
        self_len = torch.matmul(full_set, self.powers[self.T[:self.n_taxa, :self.n_taxa]] * self.d)
        return  torch.matmul(self_len, full_set)

    def h_len(self, h):
        self_len = torch.matmul(h.to(torch.float64), self.powers[self.T[:self.n_taxa, :self.n_taxa]] * self.d)
        return torch.matmul(self_len, h.to(torch.float64))

    def move(self, selected_move, a_side_idx, adj_mat):
        x = selected_move[0]
        b = selected_move[1]
        a = self.neighbors[x, a_side_idx]
        x_adj = self.set_to_adj[x]
        b_adj = self.set_to_adj[b]
        a_adj = self.set_to_adj[a]

        # detach  a and x
        x_a_other_neighbor = self.neighbors[x, 1 - a_side_idx]
        x_a_other_neighbor_idx = self.set_to_adj[self.neighbors[x, 1 - a_side_idx]]

        adj_mat[x_a_other_neighbor_idx[0], x_a_other_neighbor_idx[1]] = \
            adj_mat[x_a_other_neighbor_idx[1], x_a_other_neighbor_idx[0]] = 0
        adj_mat[a_adj[0], a_adj[1]] = adj_mat[a_adj[1], a_adj[0]] = 0

        # self.plot_phylogeny()

        # detach b
        adj_mat[b_adj[0], b_adj[1]] = adj_mat[b_adj[1], b_adj[0]] = 0
        # self.plot_phylogeny()

        # reattach a
        adj_mat[x_a_other_neighbor_idx[1], a_adj[1]] = adj_mat[ a_adj[1], x_a_other_neighbor_idx[1]] = 1
        self.plot_phylogeny(adj_mat)
        self.adj_to_set[x_a_other_neighbor_idx[1], a_adj[1]] = a
        self.adj_to_set[a_adj[1], x_a_other_neighbor_idx[1]] = self.adj_to_set[a_adj[1], a_adj[0]]
        self.set_to_adj[a] = torch.tensor([x_a_other_neighbor_idx[1], a_adj[1]], device=self.device)
        self.set_to_adj[self.adj_to_set[a_adj[1], a_adj[0]]] = torch.tensor([a_adj[1], x_a_other_neighbor_idx[1]], device=self.device)

        # reattach x
        adj_mat[b_adj[0], x_adj[0]] = adj_mat[x_adj[0], b_adj[0]] = 1
        self.adj_to_set[b_adj[0], x_adj[0]] = x
        self.adj_to_set[x_adj[0], b_adj[0]] = self.adj_to_set[x_adj[1], x_adj[0]]
        self.set_to_adj[x] = torch.tensor([x_adj[0], b_adj[0]], device=self.device)
        self.set_to_adj[self.adj_to_set[x_adj[1], x_adj[0]]] = torch.tensor([x_adj[0], b_adj[0]], device=self.device)

        self.plot_phylogeny(adj_mat)

        # reattach b
        adj_mat[b_adj[1], x_adj[0]] = adj_mat[x_adj[0], b_adj[1]] = 1
        self.adj_to_set[b_adj[1], x_adj[0]] = b
        self.adj_to_set[x_adj[0], b_adj[1]] = self.adj_to_set[b_adj[1], b_adj[0]]
        self.set_to_adj[b] = torch.tensor([x_adj[0], b_adj[1]], device=self.device)
        self.set_to_adj[self.adj_to_set[b_adj[1], b_adj[0]]] = torch.tensor([x_adj[0], b_adj[1]], device=self.device)

        # self.plot_phylogeny(adj_mat)

        # mettere nuovo subtree da a a b

        return adj_mat

    def update_sub_mat(self, selected_move):
        x = self.subtrees_mat[selected_move[0]]
        including = torch.matmul(self.subtrees_mat, x) - x.sum() >= 0
        including[selected_move[0]] = False
        self.subtrees_mat[including] -= x
        ppp = torch.nonzero(including).flatten()
        including_idxs = self.set_to_adj[torch.nonzero(including).flatten()]
        o = including_idxs
        including_c = self.adj_to_set[including_idxs[:, 1], including_idxs[:, 0]]
        self.subtrees_mat[including_c] += x
        p=0
#ll

torch.set_printoptions(precision=2, linewidth=150)

random.seed(0)
np.random.seed(0)

n = 6

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T
np.fill_diagonal(d, 0)

device = 'cpu'
# device = 'cuda:0'

model = PrecomputeTorch3(d, device=device)
t = time.time()

model.solve()
# print(model.set_to_adj)
model.plot_phylogeny()

# print(model.intersection.sum(), model.intersection.shape[0]**2)

s_move, side = model.spr()
print(time.time() - t)

model.update_sub_mat(s_move)
new_adj = model.move(s_move, side, model.solution.clone())
T = model.get_tau(new_adj)
print('new obj', (model.powers[torch.tensor(T)]*model.d).sum().item())
print(model.set_to_adj[s_move[0]], model.set_to_adj[s_move[1]])

model.solution = new_adj

model.plot_phylogeny()






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