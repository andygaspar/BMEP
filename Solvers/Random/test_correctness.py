import numpy as np
import torch


class Tester:

    def __init__(self, d, powers):
        self.d = d
        self.n_taxa = self.d.shape[0]
        self.powers = powers

    def check_dist(self, T, subtrees_mat, subtrees_dist):
        val = self.d * self.powers[T[:self.n_taxa, :self.n_taxa]]
        dist = torch.zeros_like(subtrees_dist)
        for i in range(subtrees_dist.shape[0]):
            for j in range(subtrees_dist.shape[0]):
                a = subtrees_mat[i, :self.n_taxa].float()
                b = subtrees_mat[j, :self.n_taxa].float()
                idx = torch.matmul(a.unsqueeze(0).T, b.unsqueeze(0)).to(torch.long)==1
                v = val[idx].flatten().sum() * (1 - (subtrees_mat[i] * subtrees_mat[j]).sum()>0)*2
                dist[i, j] = v
        return dist

    def test_spr(self, adj_mat, T, intersections, a_side_idx, diff_tree, set_to_adj,
                                           adj_to_set, neighbors,subtrees_mat, subtrees_dist):

        T_test = self.get_full_tau_tensor(adj_mat, self.n_taxa)
        if not torch.equal(T, T_test):
            raise  Exception("T and T_new mismatching")


        inter_neighbor = intersections * intersections[neighbors[:, a_side_idx]]
        regrafts = torch.nonzero(inter_neighbor)

        diff = torch.zeros_like(diff_tree)
        for i in range(regrafts.shape[0]):
            obj_val = (self.powers[T_test[:self.n_taxa, :self.n_taxa]] * self.d).sum()
            new_adj = self.move(regrafts[i], a_side_idx, adj_mat.clone(), neighbors, set_to_adj)
            T_new = self.get_full_tau_tensor(new_adj, self.n_taxa)[:self.n_taxa, :self.n_taxa]
            diff[i] = obj_val - (self.powers[T_new] * self.d).sum()

        print(' spr computation ', torch.allclose(diff, diff_tree))
        # for i in range(regrafts.shape[0]):
        #     print(diff_tree[i].item(), diff[i].item(), regrafts[i]) #, diff_new[i])

    def test_diff(self, selected_move, a_side_idx, T, set_to_adj, adj_to_set, neighbors, subtrees_mat, subtrees_dist):
        x = selected_move[0]
        x_adj = set_to_adj[x]
        x_c = adj_to_set[x_adj[1], x_adj[0]]

        b = selected_move[1]
        b_adj = set_to_adj[b]
        b_c = adj_to_set[b_adj[1], b_adj[0]]

        a = neighbors[x, a_side_idx]
        a_adj = set_to_adj[a]
        a_c = adj_to_set[a_adj[1], a_adj[0]]

        ab = subtrees_dist[a, b]


        #test
        h =  1 - subtrees_mat[x] - subtrees_mat[b] - subtrees_mat[a]
        h = h[:self.n_taxa]

        hx = self.h_dist(x, h, subtrees_mat=subtrees_mat, T=T)
        ah = self.h_dist(a, h, subtrees_mat=subtrees_mat, T=T)
        hb = self.h_dist(b, h, subtrees_mat=subtrees_mat, T=T)

        # LT = L(XA) + L(XB) + L(XB)
        xb = subtrees_dist[x, b] #ok
        dist_xb = self.subset_dist(x, b, subtrees_mat=subtrees_mat, T=T)

        xa = subtrees_dist[x, a] #ok
        dist_xa = self.subset_dist(x, b, subtrees_mat=subtrees_mat, T=T)

        p = set_to_adj[x][0]
        pp = set_to_adj[b][0]
        dist_ij = T[set_to_adj[x][0], set_to_adj[b][0]]
        diff_xb = xb * (1 - 1/self.powers[dist_ij])
        diff_xa = xa * (1 - self.powers[dist_ij])

        # diff_bh = bh(1 - 1/2),  bh = b<->b_c - xb - ba
        diff_bh = (subtrees_dist[b, b_c] - xb - subtrees_dist[a, b])/2
        bh = subtrees_dist[b, b_c] - xb - subtrees_dist[a, b]
        k = bh.item()
        kk = hb.item()

        # diff_ah = ah(1 - 2), ah = a<->a_c - ab - xa
        diff_ah = -(subtrees_dist[a, a_c] - subtrees_dist[a, b] - xa)


        # xh = x<->x_c - xa - xb
        xh = subtrees_dist[x, x_c] - xa - xb
        x_set_idx = torch.nonzero(subtrees_mat[x, :self.n_taxa]).flatten()

        l = self.subset_len(a, subtrees_mat, T) \
            + self.subset_len(b, subtrees_mat, T) + self.subset_len(x,subtrees_mat, T) + self.h_len(h, T) \
            + ab + xa + ah + xh + xb + hb
        l = l.item()

        p1 = self.powers[T[x_adj[0], :self.n_taxa]]
        p2 = self.powers[T[b_adj[1], :self.n_taxa]]

        diff_xh = (xh * ((p1 - p2) / p1) * h).sum(dim=-1)
        # diff_xh = xh - (self.powers[T[b_adj[1], :self.n_taxa]] * self.d[x_set_idx] * h).sum()
        diff_T = (diff_xb + diff_xa + diff_bh + diff_xh + diff_ah).item()
        # -0.06689979168715987
        return diff_T, diff_xh.item()

    def h_dist(self, subtree, h, subtrees_mat, T):
        vw = torch.matmul(subtrees_mat[subtree, :self.n_taxa].to(torch.float64),
                          self.powers[T[:self.n_taxa, :self.n_taxa]] * self.d)
        return torch.matmul(vw, h.to(torch.float64)) * 2

    def subset_dist(self, set_v, set_w, subtrees_mat, T):
        vw = torch.matmul(subtrees_mat[set_v, :self.n_taxa].to(torch.float64),
                            self.powers[T[:self.n_taxa, :self.n_taxa]] * self.d)
        return torch.matmul(vw, subtrees_mat[set_w, :self.n_taxa].to(torch.float64)) * 2

    def h_len(self, h, T):
        self_len = torch.matmul(h.to(torch.float64), self.powers[T[:self.n_taxa, :self.n_taxa]] * self.d)
        return torch.matmul(self_len, h.to(torch.float64))

    def subset_len(self, s, subtrees_mat, T):
        return self.subset_dist(s, s, subtrees_mat, T)/2

    def tree_len(self, T):
        full_set = torch.ones(self.n_taxa, dtype=torch.float64)
        self_len = torch.matmul(full_set, self.powers[T[:self.n_taxa, :self.n_taxa]] * self.d)
        return  torch.matmul(self_len, full_set)

    @staticmethod
    def move(selected_move, a_side_idx, adj_mat, neighbors, set_to_adj):
        x = selected_move[0].clone()
        b = selected_move[1].clone()
        a = neighbors[x, a_side_idx].clone()
        x_neighbor = neighbors[x, 1 - a_side_idx]

        x_adj = set_to_adj[x].clone()
        b_adj = set_to_adj[b].clone()
        a_adj = set_to_adj[a].clone()
        x_neighbor_idx = set_to_adj[x_neighbor].clone()


        # detach  a and x

        adj_mat[x_neighbor_idx[0], x_neighbor_idx[1]] = \
            adj_mat[x_neighbor_idx[1], x_neighbor_idx[0]] = 0
        adj_mat[a_adj[0], a_adj[1]] = adj_mat[a_adj[1], a_adj[0]] = 0

        # detach b
        adj_mat[b_adj[0], b_adj[1]] = adj_mat[b_adj[1], b_adj[0]] = 0

        # reattach a
        adj_mat[x_neighbor_idx[1], a_adj[1]] = adj_mat[ a_adj[1], x_neighbor_idx[1]] = 1

        # reattach x
        adj_mat[b_adj[0], x_adj[0]] = adj_mat[x_adj[0], b_adj[0]] = 1

        # reattach b
        adj_mat[b_adj[1], x_adj[0]] = adj_mat[x_adj[0], b_adj[1]] = 1


        return adj_mat

    def check_tau(self, adj_mat, Tau):
        T = self.get_full_tau_tensor(adj_mat, self.n_taxa).to(torch.long)
        equal = torch.equal(T, Tau)
        print('tau check', equal)
        if not equal:
            print(torch.nonzero(Tau - T))

            raise Exception("Tau test failed")



    @staticmethod
    def get_full_tau_tensor(adj_mat, n_taxa):
        Tau = torch.full_like(adj_mat, n_taxa)
        Tau[adj_mat > 0] = 1
        diag = torch.eye(adj_mat.shape[1]).bool()
        Tau[diag] = 0  # diagonal elements should be zero
        for i in range(adj_mat.shape[1]):
            # The second term has the same shape as Tau due to broadcasting
            Tau = torch.minimum(Tau, Tau[ i, :].unsqueeze(0)
                                + Tau[:, i].unsqueeze(1))
        return Tau.to(torch.long)