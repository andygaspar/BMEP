import random
import time

import numpy as np
import torch

from FastME.fast_me import FastMeSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.node_torch import NodeTorch

import sys

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Solvers.UCTSolver.utc_utils import nni_landscape
from Solvers.UCTSolver.utils_rollout import swa_policy
from Solvers.UCTSolver.utils_scores import max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 40

runs = 30

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 0

for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)

    mcts = UtcSolverTorch(d, swa_policy, max_score_normalised)
    mcts.solve(20)
    print(mcts.obj_val)
    t = time.time()
    sol = mcts.solution
    obj_val = mcts.obj_val
    for _ in range(5):
        expl_trees = nni_landscape(sol, mcts.n_taxa, mcts.m)
        obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).to(mcts.device), mcts.n_taxa)
        new_obj_val = torch.min(obj_vals).item()
        sol = expl_trees[torch.argmin(obj_vals)]
        if obj_val > new_obj_val:
            obj_val = new_obj_val
            print(obj_val)
    print(obj_val,  'time', time.time() - t)

    mcts_1 = UtcSolverTorch(d, swa_policy, max_score_normalised)
    mcts_1.solve(10)
    print(mcts_1.obj_val)
    t = time.time()
    expl_trees = nni_landscape(mcts_1.solution, mcts_1.n_taxa, mcts_1.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts_1.d).to(mcts.device), mcts_1.n_taxa)
    print(torch.min(obj_vals), 'time', time.time() - t)

