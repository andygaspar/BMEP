import random

import numpy as np
import pandas as pd
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.FastME.pharser_newik.newwik_handler import get_adj_from_nwk, compute_newick
from Solvers.NJ_ILP.nj_ilp import NjIlp
from Solvers.RandomNni.random_nni import RandomNni
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchBackTrack
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorchBackTrack2
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, mixed_policy, swa_nni_policy, random_policy
from Solvers.UCTSolver.utils.utils_scores import average_score_normalised, max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 30

runs = 10

results = np.zeros((runs, 4))

random.seed(0)
np.random.seed(0)
iterations = 20

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

    rand_nni = RandomNni(d)
    rand_nni.solve_timed(100)
    print("rand", rand_nni.time, rand_nni.obj_val)


    swa_nni = SwaSolverTorchNni(d)
    swa_nni.solve_timed(3, None, 10, 20,  5, 20)
    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, init_topology=swa_nni.T,
                        triangular_inequality=False, logs=False)
    fast.solve_timed()
    swa_nni.obj_val = fast.obj_val

    mcts = UtcSolverTorchBackTrack2(d, swa_policy, max_score_normalised, nni_iterations=10, nni_tol=0.02)
    mcts.solve_timed(iterations)
    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, init_topology=mcts.T,
                        triangular_inequality=False, logs=False)
    fast.solve_timed()
    mcts.obj_val = fast.obj_val

    # nj_i = NjIlp(d)
    # nj_i.solve(int(np.ceil(mcts.time)))
    # print(mcts.obj_val, nj_i.obj_val)

    mcts_ = UtcSolverTorchBackTrack(d, random_policy, max_score_normalised, nni_iterations=20, nni_tol=0.02)
    mcts_.solve_timed(2)
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
    print(swa.obj_val, swa_nni.obj_val, mcts.obj_val, fast.obj_val)
    print(mcts.time, mcts_.time, mcts_t.time, fast.time, '\n')
    #
    # fast = FastMeSolver(d, bme=True, nni=False, digits=17, post_processing=False, triangular_inequality=False, logs=False)
    # fast.solve_timed()
    # adj_mat = torch.tensor(fast.solution, dtype=torch.float64).to(fast.device)
    # dd = torch.from_numpy(fast.d).to(fast.device)
    #
    # res = run_nni_search(10, adj_mat, fast.obj_val, dd, fast.n_taxa, fast.m, fast.device)
    #
    # results.append([mcts_.obj_val, fast.obj_val])

    # print(mcts.time, mcts_.time, mcts_t.time)
    # print(fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, fast4.obj_val, "\n\n")

# results = np.array(results)
# df = pd.DataFrame({"mcts": results[:, 0], "lp_nj": results[:, 1], "fast": results[:, 2]})
# df.to_csv("test_new.csv", index_label=False, index=False)


# import pandas as pd
#
# df = pd.read_csv("testino20.csv")