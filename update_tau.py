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


def get_tau_tensor(adj_mat, n_taxa):
    sub_adj = adj_mat[n_taxa:, n_taxa:]
    Tau = torch.full_like(sub_adj, n_taxa)
    Tau[sub_adj > 0] = 1
    diag = torch.eye(sub_adj.shape[1]).bool()
    Tau[diag] = 0  # diagonal elements should be zero
    for i in range(sub_adj.shape[1]):
        # The second term has the same shape as Tau due to broadcasting
        Tau = torch.minimum(Tau, Tau[i, :].unsqueeze(0)
                            + Tau[:, i].unsqueeze(1))
    return Tau[:n_taxa, :n_taxa]


distances = DistanceData()
distances.print_dataset_names()
data_set = distances.get_dataset(3)


dim = 10

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
adj_mat = torch.tensor(solver.initial_adj_mat())
for i in range(3, solver.n_taxa):
    idxs_list = torch.nonzero(torch.triu(adj_mat))
    idxs = random.choice(idxs_list)
    adj_mat = solver.add_node(adj_mat, idxs, i, solver.n_taxa)
    T = adj_mat[dim:, dim:]
    for _ in range(2, i):
        g= torch.nonzero(T[i - 2])
        a[g] = 1
        T[i - 2, g] = 0
        g= torch.nonzero(T[i - 2])
        b[g] = 1
        T[i - 2, g] = 0




r = time.time()
t = get_tau_tensor(adj_mat, solver.n_taxa)
idxs = torch.nonzero(adj_mat[:solver.n_taxa])[:, 1]
a = torch.cartesian_prod(idxs, idxs) - solver.n_taxa
tau = (t[a[:, 0], a[:, 1]] + 2).reshape(solver.n_taxa, solver.n_taxa)
tau[range(solver.n_taxa), range(solver.n_taxa)] = 0
print(time.time() - r)




r = time.time()
tau_tens = solver.get_tau_tensor(adj_mat, solver.n_taxa)
print(time.time() - r)





print(torch.equal(tau, tau_tens))




