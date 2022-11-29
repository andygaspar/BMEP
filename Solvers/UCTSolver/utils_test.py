import torch

from Solvers.UCTSolver.utc_utils import nni_landscape

adj_mat = torch.zeros((14, 14))
n_taxa = 8
mat_size = 14

adj_mat[0, 8] = 1
adj_mat[1, 8] = 1
adj_mat[2, 13] = 1
adj_mat[3, 11] = 1
adj_mat[4, 9] = 1
adj_mat[5, 13] = 1
adj_mat[6, 12] = 1
adj_mat[7, 12] = 1
adj_mat[8, 0] = 1
adj_mat[8, 1] = 1
adj_mat[8, 9] = 1
adj_mat[9, 4] = 1
adj_mat[9, 8] = 1
adj_mat[9, 10] = 1
adj_mat[10, 9] = 1
adj_mat[10, 11] = 1
adj_mat[10, 13] = 1
adj_mat[11, 3] = 1
adj_mat[11, 10] = 1
adj_mat[11, 12] = 1
adj_mat[12, 6] = 1
adj_mat[12, 7] = 1
adj_mat[12, 11] = 1
adj_mat[13, 2] = 1
adj_mat[13, 5] = 1
adj_mat[13, 10] = 1


trees = nni_landscape(adj_mat, n_taxa, mat_size)
print(torch.all(trees[:, :n_taxa, :].sum(dim=-1) == 1))
print(torch.all(trees[:, n_taxa:, :].sum(dim=-1) == 3))

