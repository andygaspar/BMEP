import random

import numpy as np
import pandas as pd

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.NJ_ILP.nj_ilp import NjIlp
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchBackTrack
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, mixed_policy
from Solvers.UCTSolver.utils.utils_scores import average_score_normalised, max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 20

runs = 30

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 300

results = []

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
    print(swa.time, swa.obj_val)

    swa_nni = SwaSolverTorchNni(d)
    swa_nni.solve_timed(3, None, 10, 20,  5, 20)
    print(swa_nni.time, swa_nni.obj_val)

    mcts = UtcSolverTorchBackTrack(d, swa_policy, max_score_normalised, nni_iterations=10, nni_tol=0.02)
    mcts.solve_timed(iterations)
    print(mcts.time, mcts.obj_val)

    nj_i = NjIlp(d)
    nj_i.solve(int(np.ceil(mcts.time)))
    print(mcts.obj_val, nj_i.obj_val)

    mcts_ = UtcSolverTorchBackTrack(d, swa_policy, max_score_normalised, nni_iterations=10, nni_tol=0.02)
    mcts_.solve_timed(iterations*2)
    print(mcts_.time, mcts_.obj_val)
    # mcts_.solve_timed(iterations)
    # print(mcts_.obj_val)
    # improved, nni_val, nni_sol = \
    #     run_nni_search(100, mcts_.solution, mcts_.obj_val, torch.tensor(mcts_.d).to(mcts_.device), mcts_.n_taxa,
    #                    mcts_.m, mcts_.device)
    # print("mmm ", nni_val)

    mcts_t = UtcSolverTorch(d, mixed_policy, average_score_normalised)
    # mcts_t.solve_timed(iterations)
    #

    #
    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False, logs=False)
    fast.solve_timed()
    print(mcts.obj_val, mcts_.obj_val, mcts_t.obj_val, fast.obj_val)
    print(mcts.time, mcts_.time, mcts_t.time, fast.time, '\n')

    results.append([mcts_.obj_val, nj_i, fast.obj_val])

    # print(mcts.time, mcts_.time, mcts_t.time)
    # print(fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, fast4.obj_val, "\n\n")

results = np.array(results)
df = pd.DataFrame({"mcts": results[:, 0], "lp_nj": results[:, 1], "fast": results[:, 2]})
df.to_csv("test_new.csv", index_label=False, index=False)


# import pandas as pd
#
# df = pd.read_csv("testino20.csv")