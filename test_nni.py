import random

import numpy as np
import torch

from FastME.fast_me import FastMeSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.UCTSolver.utc_solver import UtcSolver

import sys

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorch_1
from Solvers.UCTSolver.utc_utils import nni_landscape

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 40

runs = 1

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 0

for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)



    mcts = UtcSolverTorch(d)
    mcts.solve(10)
    print(mcts.obj_val)
    expl_trees = nni_landscape(mcts.solution, mcts.n_taxa, mcts.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).cuda(), mcts.n_taxa)
    print(torch.min(obj_vals))

    mcts_1 = UtcSolverTorch(d)
    mcts_1.solve(15)
    print(mcts_1.obj_val)
    expl_trees = nni_landscape(mcts_1.solution, mcts_1.n_taxa, mcts_1.m)
    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts_1.d).cuda(), mcts_1.n_taxa)
    print(torch.min(obj_vals))

