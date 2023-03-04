import random
import time
import numpy as np
import torch

from Solvers.Random.test_correctness import Tester
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
        self.adj_mat = None

        self.subtrees_mat = None
        self.pointer_subtrees = None
        self.adj_to_set = None
        self.subtree_dist = None
        self.T_new = None

        self.tester = Tester(self.d, self.powers)

        self.mask = torch.zeros(self.n_subtrees, dtype=torch.bool, device=self.device)

    def solve(self, start=3, adj_mat=None, test=False):
        self.init_tree()
        self.plot_phylogeny(self.adj_mat)
        for i in range(self.subtrees_mat.shape[0]):
            print(self.set_to_adj[i], self.set_to_adj[self.neighbors[i][0]], self.set_to_adj[self.neighbors[i][1]],
                  self.subtrees_mat[i])
        to_be_continued = True
        while to_be_continued:
            s_move, side, to_be_continued = self.spr(test)
            # self.plot_phylogeny(self.adj_mat)
            print(self.set_to_adj[s_move[0]], '->', self.set_to_adj[s_move[1]])

            # self.update_tau(s_move, side)
            if to_be_continued:

                self.update_sub_mat(s_move)
                new_adj = self.move(s_move, side, self.adj_mat.clone())
                self.adj_mat = new_adj
                # self.T = self.update_tau(self.adj_mat).to(torch.long)
                self.T = self.get_full_tau_tensor(self.adj_mat, self.n_taxa).to(torch.long)
                s = self.subtrees_mat[:, :self.n_taxa].to(torch.float64)
                self.subtree_dist = self.compute_subtrees_dist(s)
                self.intersection = torch.matmul(s, s.T) == 0
                self.subtree_dist *= self.intersection
                self.neighbors = self.compute_neighbors(self.set_to_adj, self.adj_to_set, self.adj_mat)
                self.plot_phylogeny(self.adj_mat)


                print(self.compute_obj_tensor().item())
            else:
                to_be_continued = False

        self.solution = self.adj_mat

    def update_tau(self, adj_mat):
        return self.get_full_tau_tensor(adj_mat, self.n_taxa)
    def init_tree(self):
        t = time.time()
        adj_mat = self.initial_adj_mat(self.device)
        subtrees_mat, adj_to_set  = self.initial_sub_tree_mat()
        T = self.init_tau()
        s_mat_step = 6
        # subtree_dist = self.init_sub_dist()
        for step in range(3, self.n_taxa):
            choices = 3 + (step - 3) * 2
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            rand_idxs = random.choices(range(choices))
            idxs_list = idxs_list[rand_idxs][0]
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)

            subtrees_mat, adj_to_set, T = \
                self.add_subtrees(subtrees_mat, adj_to_set, step, idxs_list, s_mat_step, T)
            s_mat_step += 4

        s = subtrees_mat[:, :self.n_taxa].to(torch.float64)
        self.T = T
        print('build time', time.time() - t)
        self.subtree_dist = self.compute_subtrees_dist(s)
        self.intersection = torch.matmul(s, s.T) == 0
        self.subtree_dist *= self.intersection
        self.subtrees_mat = subtrees_mat
        self.adj_to_set = adj_to_set
        self.adj_mat = adj_mat

        idxs = torch.nonzero(self.adj_mat).T
        order = torch.argsort(self.adj_to_set[idxs[0], idxs[1]])
        self.set_to_adj = idxs.T[order]
        print('s dist', time.time() - t)
        self.neighbors = self.compute_neighbors(self.set_to_adj, self.adj_to_set, self.adj_mat)
        # self.device = 'cuda:0'
        print('full build time', time.time() - t)
        self.obj_val = (self.powers[self.T[:self.n_taxa, :self.n_taxa]]*self.d).sum().item()
        print('obj val', self.obj_val)
        # t = time.time()

        # print(time.time() - t)

    def compute_subtrees_dist(self, s):
        return torch.matmul(torch.matmul(s, self.d * self.powers[self.T[:self.n_taxa, :self.n_taxa]]), s.T) * 2

    def initial_sub_tree_mat(self):
        subtree_mat = torch.zeros((self.n_subtrees, self.m), device=self.device, dtype=torch.long)
        adj_to_set = -torch.ones((self.m, self.m), device=self.device, dtype=torch.long)

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
        mask_j[range(self.n_subtrees), j] = 1

        attached_to_i = torch.nonzero(adj[i] - mask_j)

        conversion = torch.zeros_like(attached_to_i)
        k = set_to_adj[attached_to_i[:, 0], 0]
        conversion[:, 0] = k
        conversion[:, 1] = attached_to_i[:, 1]

        final_conversion = torch.stack([attached_to_i[:, 0], adj_to_set[conversion[:, 0], conversion[:, 1]]]).T

        neighbors = torch.zeros((self.n_subtrees, 2), dtype=torch.long, device=self.device)
        f = final_conversion.view((final_conversion.shape[0] // 2, 2, 2))
        neighbors[f[:, 0, 0], 0] = f[:, 0, 1]
        neighbors[f[:, 1, 0], 1] = f[:, 1, 1]

        # print(neighbors)

        return neighbors


    def spr(self, test):
        # intersections = self.intersection.clone()
        intersections = self.intersection

        # distance subtrees to taxa, useful for h computation
        sub_to_taxon_idx = torch.nonzero(self.adj_mat[:, :self.n_taxa]).T
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
        intersections[:, self.subtrees_mat.sum(dim=-1) == self.m - 1] = \
            intersections[self.subtrees_mat.sum(dim=-1) == self.m - 1] = False

        # neighbor 1
        a_side = 0
        max_first_side_val, max_first_side_move, diff_T = self.spr_side(intersections, sub_to_taxon, a_side, test)

        if test:
            self.tester.test_spr(self.adj_mat, self.T, intersections.clone(), a_side, self.obj_val, diff_T, self.set_to_adj,
                                 self.adj_to_set, self.neighbors, self.subtrees_mat, self.subtree_dist)


        # neightbor 2
        a_side = 1
        max_second_side_val, max_second_side_move, diff_T = self.spr_side(intersections, sub_to_taxon, a_side, test)


        max_val, max_side = torch.max(torch.stack([max_first_side_val, max_second_side_val], dim=-1), 0) # a side idx
        best_move = torch.cat([max_first_side_move.unsqueeze(0), max_second_side_move.unsqueeze(0)])

        if test:
            self.tester.test_spr(self.adj_mat, self.T, intersections, a_side, self.obj_val, diff_T, self.set_to_adj,
                                 self.adj_to_set, self.neighbors, self.subtrees_mat, self.subtree_dist)

        return best_move[max_side], max_side, max_val.item() > 0

    def spr_side(self, intersections, sub_to_taxon, a_side, test):

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

        if not test:
            diff_T = None

        return max_side_val, max_side_move, diff_T

    def move(self, selected_move, a_side_idx, adj_mat):
        x = selected_move[0].clone()
        b = selected_move[1].clone()
        a = self.neighbors[x, a_side_idx].clone()
        x_neighbor = self.neighbors[x, 1 - a_side_idx]

        x_adj = self.set_to_adj[x].clone()
        b_adj = self.set_to_adj[b].clone()
        a_adj = self.set_to_adj[a].clone()
        x_neighbor_idx = self.set_to_adj[x_neighbor].clone()

        x_c = self.adj_to_set[x_adj[1], x_adj[0]].clone()
        b_c = self.adj_to_set[b_adj[1], b_adj[0]].clone()
        a_c = self.adj_to_set[a_adj[1], a_adj[0]].clone()
        x_c_neighbor = self.adj_to_set[x_neighbor_idx[1], x_neighbor_idx[0]].clone()

        # detach  a and x

        new_subtree_root = b_adj[0].clone()

        adj_mat[x_neighbor_idx[0], x_neighbor_idx[1]] = \
            adj_mat[x_neighbor_idx[1], x_neighbor_idx[0]] = 0
        adj_mat[a_adj[0], a_adj[1]] = adj_mat[a_adj[1], a_adj[0]] = 0


        # detach b
        adj_mat[b_adj[0], b_adj[1]] = adj_mat[b_adj[1], b_adj[0]] = 0

        # reattach a
        adj_mat[x_neighbor_idx[1], a_adj[1]] = adj_mat[ a_adj[1], x_neighbor_idx[1]] = 1
        self.adj_to_set[x_neighbor_idx[1], a_adj[1]] = a
        self.adj_to_set[a_adj[1], x_neighbor_idx[1]] = a_c
        self.set_to_adj[a] = torch.tensor([x_neighbor_idx[1], a_adj[1]], device=self.device)
        self.set_to_adj[a_c] = torch.tensor([a_adj[1], x_neighbor_idx[1]], device=self.device)

        # reattach x
        adj_mat[b_adj[0], x_adj[0]] = adj_mat[x_adj[0], b_adj[0]] = 1

        # reattach b
        adj_mat[b_adj[1], x_adj[0]] = adj_mat[x_adj[0], b_adj[1]] = 1
        self.adj_to_set[x_adj[0], b_adj[1]] = b
        self.adj_to_set[b_adj[1], x_adj[0]] = b_c
        self.set_to_adj[b] = torch.tensor([x_adj[0], b_adj[1]], device=self.device)
        self.set_to_adj[b_c] = torch.tensor([b_adj[1], x_adj[0]], device=self.device)

        # new subtree (x + b) ad its comp stored in x_a_other neighbor and its comp
        self.adj_to_set[new_subtree_root, x_adj[0]] = x_neighbor
        self.adj_to_set[x_adj[0], new_subtree_root] = x_c_neighbor
        self.set_to_adj[x_neighbor] = torch.tensor([new_subtree_root, x_adj[0]], device=self.device)
        self.set_to_adj[x_c_neighbor] = torch.tensor([x_adj[0], new_subtree_root], device=self.device)


        x = self.subtrees_mat[selected_move[0]].clone()
        x[x_adj[0]] = 1
        b = self.subtrees_mat[selected_move[1]]

        # new sets
        p = self.adj_to_set[x_adj[0], b_adj[0]]
        pp = self.adj_to_set[b_adj[0], x_adj[0]]
        self.subtrees_mat[self.adj_to_set[x_adj[0], b_adj[0]]] = 1 - x - b
        self.subtrees_mat[self.adj_to_set[b_adj[0], x_adj[0]]] = x + b
        self.plot_phylogeny(adj_mat)

        for i in range(self.subtrees_mat.shape[0]):
            idx_ = self.set_to_adj[i]
            print(self.set_to_adj[i], self.adj_to_set[idx_[0], idx_[1]], self.subtrees_mat[i])


        return adj_mat

    def update_sub_mat(self, selected_move):

        x = selected_move[0]
        x_subset = self.subtrees_mat[x].clone()
        x_subset[self.set_to_adj[x][0]] = 1 # adding x internal node
        x_idx = self.set_to_adj[x]

        b = selected_move[1]
        b_subset = self.subtrees_mat[b].clone()
        b_subset[self.set_to_adj[b][0]] = 1 # adding b internal node
        b_idx = self.set_to_adj[b]
        b_c = self.adj_to_set[b_idx[1], b_idx[0]]

        including_x = torch.matmul(self.subtrees_mat.to(torch.float64), x_subset.to(torch.float64)) - x_subset.sum() == 0
        including_b = torch.matmul(self.subtrees_mat.to(torch.float64), b_subset.to(torch.float64)) - b_subset.sum() == 0


        x_side_trees = including_x * self.intersection[b]
        x_side_trees[b_c] = True
        b_side_trees = including_b * self.intersection[x]
        b_side_trees[b_c] = True
        self.subtrees_mat[x_side_trees, :] -= x_subset
        self.subtrees_mat[b_side_trees] += x_subset

    def update_tau_(self, selected_move, a_side_idx):
        x = selected_move[0]
        b = selected_move[1]
        a = self.neighbors[x, a_side_idx]

        x_adj = self.set_to_adj[x].clone()
        b_adj = self.set_to_adj[b].clone()
        a_adj = self.set_to_adj[a].clone()

        x_sub = self.subtrees_mat[x]
        b_sub = self.subtrees_mat[b]
        a_sub = self.subtrees_mat[a]

        a_in, a_out = a_sub == 1, a_sub == 0
        self.T[a_in, a_out] -= 1
        self.T[a_out, a_in] -= 1

        b_in, b_out = b_sub == 1, b_sub == 0
        self.T[b_in, b_out] += 1
        self.T[b_out, b_in] += 1

        x_in, x_out = x_sub == 1, x_sub == 0

        p = self.T[x_in, x_adj[0]]
        pp = self.T[b_adj[0], x_out]
        # inner dist to self root + from b root anywhere else
        self.T[x_in, x_out] = self.T[x_out, x_in] = self.T[x_in, x_adj[0]] + self.T[b_adj[0], x_out] + 1
        print(self.T)


        # self.T[b_adj[0], ~mask] = self.T[~mask, b_adj[0]] =

        p = 0


#ll

torch.set_printoptions(precision=2, linewidth=150)

seed = 3
random.seed(3)
np.random.seed(3)

n = 6

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T
np.fill_diagonal(d, 0)

device = 'cpu'
# device = 'cuda:0'

model = PrecomputeTorch3(d, device=device)
t = time.time()

model.solve(test=True)

print(time.time() - t)
print('final obj', model.compute_obj_tensor().item())

