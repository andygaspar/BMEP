import random

import numpy as np
import torch

from FastME.fast_me import FastMeSolver
from Solvers.PardiSolver.pardi_solver_parallel import PardiSolverParallel
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchBackTrack
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, mixed_policy, random_policy
from Solvers.UCTSolver.utils.utils_scores import average_score_normalised, max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 15

runs = 30

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 60

for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)
    # nj_i = NjIlp(d)
    # nj_i.solve(2)
    # print(nj_i.obj_val)
    #0.26646004352783204
    # pardi = PardiSolverParallel(d)
    # pardi.solve()
    # print(pardi.obj_val, "enumeration")

    swa = SwaSolverTorch(d)
    swa.solve_timed()
    # print(swa.time, swa.obj_val)

    swa_nni = SwaSolverTorchNni(d)
    swa_nni.solve_timed(3, None, 10, 20,  5, 20)
    # print(swa_nni.time, swa_nni.obj_val)

    mcts = UtcSolverTorchBackTrack(d, mixed_policy, average_score_normalised, nni_iterations=20, nni_tol=0.02)
    mcts.solve_timed(iterations)
    # print(mcts.time, mcts.obj_val)

    mcts_ = UtcSolverTorchBackTrack(d, swa_policy, max_score_normalised, nni_iterations=10, nni_tol=0.02)
    mcts_.solve_timed(iterations*2)

    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False, logs=False)
    fast.solve_timed()
    print(swa.obj_val, mcts_.obj_val, mcts_.obj_val, fast.obj_val)
    print(swa.time, mcts_.time, mcts_.time, fast.time, '\n')

    # print(mcts.time, mcts_.time, mcts_t.time)
    # print(fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, fast4.obj_val, "\n\n")

