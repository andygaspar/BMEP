import random

import numpy as np
import torch

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchBackTrack
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, mixed_policy, random_policy
from Solvers.UCTSolver.utils.utils_scores import average_score_normalised, max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 35

runs = 1

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 100

for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)
    # nj_i = NjIlp(d)
    # nj_i.solve(2)
    # print(nj_i.obj_val)
#0.26646004352783204
    mcts = UtcSolverTorchBackTrack(d, mixed_policy, average_score_normalised, nni_iterations=10, nni_tol=0.02)
    mcts.solve_timed(iterations)
    print(mcts.time, mcts.obj_val)

    mcts_ = UtcSolverTorchBackTrack(d, mixed_policy, max_score_normalised, nni_iterations=10, nni_tol=0.02)
    # mcts_.solve_timed(iterations)
    # print(mcts_.obj_val)
    # improved, nni_val, nni_sol = \
    #     run_nni_search(100, mcts_.solution, mcts_.obj_val, torch.tensor(mcts_.d).to(mcts_.device), mcts_.n_taxa,
    #                    mcts_.m, mcts_.device)
    # print("mmm ", nni_val)

    mcts_t = UtcSolverTorch(d, mixed_policy, average_score_normalised)
    # mcts_t.solve_timed(iterations)
    #
    # print(mcts.time, mcts_.time, mcts_t.time)
    # print(mcts.obj_val, mcts_.obj_val, mcts_t.obj_val)

    # fast = FastMeSolver(d, bme=False, nni=False, triangular_inequality=False, logs=False)
    # fast.solve_timed()
    #
    # fast1 = FastMeSolver(d, bme=True, nni=False, triangular_inequality=False, logs=False)
    # fast1.solve_timed()
    #
    # fast2 = FastMeSolver(d, bme=True, nni=True, triangular_inequality=False, logs=False)
    # fast2.solve_timed()
    #
    # fast3 = FastMeSolver(d, bme=True, nni=True, digits=17, triangular_inequality=True, logs=False)
    # fast3.solve_timed()
    #
    # fast4 = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=True, logs=False)
    # fast4.solve_timed()

    # print(fast.time, fast1.time, fast2.time, fast3.time, fast4.time)
    # print(fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, fast4.obj_val, "\n\n")

