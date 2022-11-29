import random

import numpy as np
import torch

from FastME.fast_me import FastMeSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.node_torch import NodeTorch

import sys

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_utils import nni_landscape
from Solvers.UCTSolver.utils_rollout import swa_policy
from Solvers.UCTSolver.utils_scores import max_score_normalised

distances = DistanceData()
distances.print_dataset_names()
print(distances.get_dataset_names())
data_set = distances.get_dataset(3)


dim = 30

runs = 9

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 0

for run in range(runs):
    print(f'Run on dataset {run}')
    d = data_set.get_random_mat(dim)

    mcts = UtcSolverTorch(d, swa_policy, max_score_normalised)
    mcts.solve(10)
    print(f'mcts: {mcts.obj_val}')
    expl_trees = nni_landscape(mcts.solution, mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 1: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 2: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 3: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 4: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 5: {torch.min(obj_vals)}')

    mcts = UtcSolverTorch(d, swa_policy, max_score_normalised, nni=3)
    mcts.solve(10)
    print(f'mcts w/nni: {mcts.obj_val}')
    expl_trees = nni_landscape(mcts.solution, mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 1: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 2: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 3: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 4: {torch.min(obj_vals)}')
    expl_trees = nni_landscape(expl_trees[torch.argmin(obj_vals)], mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(f'nni 5: {torch.min(obj_vals)}')
