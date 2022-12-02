import random
import time

import numpy as np
import torch

from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Solvers.UCTSolver.node_torch import NodeTorch

from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Solvers.UCTSolver.utils.utc_utils import nni_landscape
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch
from Solvers.UCTSolver.utils.utils_rollout import swa_policy
from Solvers.UCTSolver.utils.utils_scores import max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 21

runs = 1

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 0

for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)

    swa = SwaSolverTorch(d)
    swa.solve_timed()
    print(swa.obj_val, "swa", swa.time)
    obj_val = swa.obj_val
    sol = torch.tensor(swa.solution)

    for _ in range(20):
        expl_trees = nni_landscape(sol, swa.n_taxa, swa.m)
        obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, swa.d, swa.n_taxa)
        new_obj_val = torch.min(obj_vals).item()
        sol = expl_trees[torch.argmin(obj_vals)]
        if obj_val > new_obj_val:
            obj_val = new_obj_val
    print(obj_val,  'last run swa')

    swa_nni = SwaSolverTorchNni(d)
    swa_nni.solve_timed(3, None, 10, 10,  5, 20)
    print(swa_nni.obj_val, "swa nni", swa_nni.time)
    obj_val = swa_nni.obj_val
    sol = torch.tensor(swa_nni.solution)

    for _ in range(20):
        expl_trees = nni_landscape(sol, swa_nni.n_taxa, swa_nni.m)
        obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, swa_nni.d, swa_nni.n_taxa)
        new_obj_val = torch.min(obj_vals).item()
        sol = expl_trees[torch.argmin(obj_vals)]
        if obj_val > new_obj_val:
            obj_val = new_obj_val
            print(obj_val)
    print(obj_val,  'last run swa nni') # 0.24215036995605468 0.24187009716705324

    #0.24363763723876952 0.2436069687878418
    new_mat = torch.cat([torch.tensor(swa.solution).unsqueeze(0), torch.tensor(swa_nni.solution).unsqueeze(0)], dim=0)

    run_nni_search_batch(2, new_mat, 3, swa.d, swa.n_taxa, swa.m,swa.device)

    # mcts = UtcSolverTorch(d, swa_policy, max_score_normalised)
    # mcts.solve(20)
    # print(mcts.obj_val)
    # t = time.time()
    # sol = mcts.solution
    # obj_val = mcts.obj_val
    # for _ in range(5):
    #     expl_trees = nni_landscape(sol, mcts.n_taxa, mcts.m)
    #     obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts.d).to(mcts.device), mcts.n_taxa)
    #     new_obj_val = torch.min(obj_vals).item()
    #     sol = expl_trees[torch.argmin(obj_vals)]
    #     if obj_val > new_obj_val:
    #         obj_val = new_obj_val
    #         print(obj_val)
    # print(obj_val,  'time', time.time() - t)
    #
    # mcts_1 = UtcSolverTorch(d, swa_policy, max_score_normalised)
    # mcts_1.solve(10)
    # print(mcts_1.obj_val)
    # t = time.time()
    # expl_trees = nni_landscape(mcts_1.solution, mcts_1.n_taxa, mcts_1.m)
    # obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, torch.from_numpy(mcts_1.d).to(mcts.device), mcts_1.n_taxa)
    # print(torch.min(obj_vals), 'time', time.time() - t)

