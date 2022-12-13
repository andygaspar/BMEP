import random
import time

import numpy as np
import pandas as pd
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.FastME.pharser_newik.newwik_handler import get_adj_from_nwk, compute_newick
from Solvers.NJ_ILP.nj_ilp import NjIlp
from Solvers.Random.random_solver import RandomSolver
from Solvers.RandomNni.random_nni import RandomNni
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchSingleBackTrack
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorchBackTrack2
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, mixed_policy, swa_nni_policy, random_policy
from Solvers.UCTSolver.utils.utils_scores import average_score_normalised, max_score_normalised
from check_nni import cn


def get_tau_tensor(adj_mat, n_taxa, device):
    sub_adj = adj_mat[:, n_taxa:, n_taxa:]
    Tau = torch.full_like(sub_adj, n_taxa)
    Tau[sub_adj > 0] = 1
    diag_idx = torch.tensor(range(n_taxa), device=device)
    Tau[:, diag_idx[:-2], diag_idx[:-2]] = 0  # diagonal elements should be zero
    for i in range(sub_adj.shape[1]):
        # The second term has the same shape as Tau due to broadcasting
        Tau = torch.minimum(Tau, Tau[:, i, :].unsqueeze(1)
                            + Tau[:, :, i].unsqueeze(2))

    idxs = torch.nonzero(adj_mat[:, :n_taxa])[:, [0, 2]]
    k = idxs[:, 1].reshape(batch_size, n_taxa).repeat(1, n_taxa).flatten()
    idxs = idxs.repeat_interleave(solver.n_taxa, 0)
    b = torch.column_stack((idxs, k))
    b[:, [1, 2]] -= n_taxa
    Tau = (Tau[b[:, 0], b[:, 1], b[:, 2]] + 2).reshape(batch_size, n_taxa, n_taxa)
    Tau[:, diag_idx, diag_idx] = 0
    return Tau


def compute_ttaus(adj_mat, n_taxa, device):
    Tau = torch.full_like(adj_mat, n_taxa)
    Tau[adj_mat > 0] = 1
    diag = torch.eye(adj_mat.shape[1], device=device).repeat(adj_mat.shape[0], 1, 1).bool()
    Tau[diag] = 0  # diagonal elements should be zero
    for i in range(adj_mat.shape[1]):
        # The second term has the same shape as Tau due to broadcasting
        Tau = torch.minimum(Tau, Tau[:, i, :].unsqueeze(1).repeat(1, adj_mat.shape[1], 1)
                            + Tau[:, :, i].unsqueeze(2).repeat(1, 1, adj_mat.shape[1]))
    return  Tau[:, :n_taxa, :n_taxa]

distances = DistanceData()
distances.print_dataset_names()
data_set = distances.get_dataset(3)


dim = 50

runs = 1

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 20

d = data_set.get_random_mat(dim)
solver = RandomSolver(d)

T = torch.zeros((8,8))
a = torch.zeros(8)
b = torch.zeros(8)
batch_size = 40
adj_mats = torch.tensor(solver.initial_adj_mat(n_problems=batch_size)).to('cpu')
for step in range(3, dim):
    choices = 3 + (step - 3) * 2
    idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch_size, -1, 3)
    rand_idxs = random.choices(range(choices), k=batch_size)
    idxs_list = idxs_list[[range(batch_size)], rand_idxs, :]
    idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
    adj_mats = solver.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=dim)
    # T = adj_mat[dim:, dim:]
    # for _ in range(2, i):
    #     g= torch.nonzero(T[i - 2])
    #     a[g] = 1
    #     T[i - 2, g] = 0
    #     g= torch.nonzero(T[i - 2])
    #     b[g] = 1
    #     T[i - 2, g] = 0


device = 'cpu'


r = time.time()
tau = get_tau_tensor(adj_mats, solver.n_taxa, device)
print(time.time() - r)




r = time.time()
tau_tens = compute_ttaus(adj_mats, solver.n_taxa, device)
print(time.time() - r)


# b = torch.vstack([idxs.repeat_interleave(solver.n_taxa), idxs.repeat(solver.n_taxa)]) - solver.n_taxa
# tau_ = (t[b[0], b[1]] + 2).reshape(solver.n_taxa, solver.n_taxa)


print(torch.equal(tau, tau_tens))

