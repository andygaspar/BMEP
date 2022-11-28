import random

import numpy as np

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_bounds import UtcSolverTorchBounds
from Solvers.UCTSolver.utils_rollout import swa_policy, random_policy, mixed_policy
from Solvers.UCTSolver.utils_scores import average_score_normalised, max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 30

runs = 1

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 0

for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)
    # nj_i = NjIlp(d)
    # nj_i.solve(2)
    # print(nj_i.obj_val)

    mcts = UtcSolverTorch(d, swa_policy, max_score_normalised)
    mcts.solve_timed(10)
    print(mcts.n_nodes)

    # swa_new = SwaSolver(d)
    # swa_new.solve_timed()

    mcts_1 = UtcSolverTorch(d, swa_policy, average_score_normalised)
    mcts_1.solve_timed(15)
    print(mcts_1.n_nodes)

    mcts_t = UtcSolverTorch(d, mixed_policy, average_score_normalised)
    mcts_t.solve_timed(150)
    print(mcts_t.n_nodes)

    mcts_t_1 = UtcSolverTorchBounds(d, mixed_policy, max_score_normalised)
    mcts_t_1.solve_timed(1000)
    print(mcts_t_1.n_nodes)

    print(mcts.time, mcts_1.time, mcts_t.time, mcts_t_1.time)
    print(mcts.obj_val, mcts_1.obj_val, mcts_t.obj_val, mcts_t_1.obj_val,
          np.argmin([mcts.obj_val, mcts_1.obj_val, mcts_t.obj_val, mcts_t_1.obj_val]))
    print('')

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

