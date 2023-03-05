import random

from Solvers.UCTSolver.utils.utils_rollout import adjust_matrices
import torch
class Subtrees:

    def __init__(self, adj_mat, subtrees_mat, n_taxa, m, device):


        self.adj_mat = adj_mat
        self.subtrees_mat = subtrees_mat
        self.n_subtrees = self.subtrees_mat.shape[0]
        self.n_taxa = n_taxa
        self.m = m
        self.device = device

        self.computed_adj_mat = None
        self.computed_subtrees_mat = None

        self.adj_to_set = None
        self.set_to_adj = None

        trajectory = self.tree_climb(adj_mat.unsqueeze(0)).flatten()
        self.computed_subtrees_mat = self.init_tree(trajectory)

        self.check_subtrees()

        # print(torch.equal(self.adj_mat, self.computed_adj_mat))
        # print(torch.equal(self.subtrees_mat, self.computed_subtrees_mat))


    def tree_climb(self, adj_mats):
        last_inserted_taxa = self.n_taxa - 1
        n_internals = self.m - self.n_taxa

        adj_mats = adjust_matrices(adj_mats, last_inserted_taxa, n_internals, self.n_taxa)
        adj_mats = adj_mats.unsqueeze(1).repeat(1, 2, 1, 1)
        reversed_idxs = torch.tensor([[i, i - 1] for i in range(1, adj_mats.shape[0] * 2, 2)],
                                     device=self.device).flatten()
        trajectories = torch.zeros((adj_mats.shape[0], self.n_taxa - 3), dtype=torch.int)

        last_inserted_taxa = self.n_taxa - 1
        for step in range(self.m - 1, self.n_taxa, -1):
            adj_mats[:, 1, :, last_inserted_taxa] = adj_mats[:, 1, last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mats[:, 1, step])
            idxs = torch.column_stack([idxs, idxs[:, 1][reversed_idxs]])

            adj_mats[idxs[:, 0], 1, idxs[:, 1], idxs[:, 2]] = adj_mats[idxs[:, 0], 1, idxs[:, 2], idxs[:, 1]] = 1
            adj_mats[:, 1, :, step] = adj_mats[:, 1, step, :] = 0

            k = (last_inserted_taxa - 1) * 2 - 1
            all_non_zeros = torch.nonzero(torch.triu(adj_mats[:, 1, :, :])).view(adj_mats.shape[0], k, 3)
            chosen = idxs[range(0, adj_mats.shape[0] * 2, 2)].repeat_interleave(k, dim=0).view(adj_mats.shape[0], k, 3)
            tj = torch.argwhere((all_non_zeros == chosen).prod(dim=-1))
            trajectories[:, last_inserted_taxa - 3] = tj[:, 1]

            last_inserted_taxa -= 1
        return trajectories

    def init_tree(self, trajectory):

        adj_mat = self.initial_adj_mat()
        subtrees_mat, adj_to_set = self.initial_sub_tree_mat()
        s_mat_step = 6
        # subtree_dist = self.init_sub_dist()
        for step in range(3, self.n_taxa):
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            idxs_list = idxs_list[trajectory[step - 3]]
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)

            subtrees_mat, adj_to_set = \
                self.add_subtrees(subtrees_mat, adj_to_set, step, idxs_list, s_mat_step)
            s_mat_step += 4


        self.computed_adj_mat = adj_mat
        self.adj_to_set = adj_to_set

        idxs = torch.nonzero(self.adj_mat).T  # use given adj_mat
        order = torch.argsort(self.adj_to_set[idxs[0], idxs[1]])
        self.set_to_adj = idxs.T[order]

        return subtrees_mat # da sistemare

    def initial_sub_tree_mat(self):
        subtree_mat = torch.zeros((self.n_subtrees, self.m), device=self.device, dtype=torch.long)
        adj_to_set = -torch.ones((self.m, self.m), device=self.device, dtype=torch.long)

        # 0
        subtree_mat[0, 0] = 1
        subtree_mat[1, 1] = 1
        subtree_mat[2, 2] = 1

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

        return subtree_mat, adj_to_set

    def add_subtrees(self, subtree_mat, adj_to_set, new_taxon_idx, idx, s_mat_step):

        new_internal_idx = self.n_taxa + new_taxon_idx - 2

        # singleton {k}
        subtree_mat[s_mat_step, new_taxon_idx] = 1
        adj_to_set[new_internal_idx, new_taxon_idx] = s_mat_step

        # {k} complementary
        subtree_mat[s_mat_step + 1, :new_taxon_idx] = 1
        subtree_mat[s_mat_step + 1, self.n_taxa: new_internal_idx + 1] = 1
        adj_to_set[new_taxon_idx, new_internal_idx] = s_mat_step + 1


        # add distance k and internal k
        ij = subtree_mat[adj_to_set[idx[0], idx[1]]]
        ji = subtree_mat[adj_to_set[idx[1], idx[0]]]

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

        return subtree_mat, adj_to_set


    @staticmethod
    def add_node(adj_mat, idxs, new_node_idx, n):
        adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
        adj_mat[idxs[0], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
        adj_mat[idxs[1], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[new_node_idx, n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat

    def initial_adj_mat(self):
        adj_mat = torch.zeros((self.m, self.m),dtype=torch.short).to(self.device)
        adj_mat[0, self.n_taxa] = adj_mat[self.n_taxa, 0] = 1
        adj_mat[1, self.n_taxa] = adj_mat[self.n_taxa, 1] = 1
        adj_mat[2, self.n_taxa] = adj_mat[self.n_taxa, 2] = 1

        return adj_mat

    def check_subtrees(self):
        equal = True
        for i in self.subtrees_mat:
            found_subtree = False
            for j in self.computed_subtrees_mat:
                if torch.equal(i[:self.n_taxa], j[:self.n_taxa]):
                    found_subtree = True
            if not found_subtree:
                print('subtree not found', i)
            equal = equal and found_subtree
        print(equal)