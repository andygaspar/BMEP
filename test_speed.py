import numpy as np

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.Random.guided_random import GuidedRandSolver
from Solvers.RandomNni.random_nni import RandomNni
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchSingleBackTrack
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorchBackTrack2
from Solvers.UCTSolver.utils.utils_rollout import random_policy
from Solvers.UCTSolver.utils.utils_scores import max_score_normalised

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 10

runs = 1

results = np.zeros((runs, 4))

# random.seed(0)
# np.random.seed(0)
iterations = 1

results = []



for run in range(runs):
    print(run)
    d = data_set.get_random_mat(dim)

    mcts_random = UtcSolverTorchBackTrack2(d, random_policy, max_score_normalised)
    mcts_random.solve_timed(iterations)
    print(mcts_random)
    it = 1

    # guided_rand = GuidedRandSolver(d, it)
    # guided_rand.solve_timed()
    # print(guided_rand.obj_val, guided_rand.time)
    # print(guided_rand.n_nodes, 'nodes', guided_rand.max_depth, ' depth', guided_rand.n_trajectories, 'tj',
    #       guided_rand.n_trees, 'trees')

    comparison = RandomNni(d, parallel=False)
    comparison.solve_timed(it)
    print(comparison.obj_val)





    swa = SwaSolverTorch(d)
    # swa.solve_timed()


    swa_nni = SwaSolverTorchNni(d)
    # swa_nni.solve_timed(3, None, 10, 20,  5, 20)
    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, init_topology=swa_nni.T,
                        triangular_inequality=False, logs=False)
    # fast.solve_timed()
    swa_nni.obj_val = fast.obj_val


    # nj_i = NjIlp(d)
    # nj_i.solve(int(np.ceil(mcts.time)))
    # print(mcts.obj_val, nj_i.obj_val)

    mcts_random = UtcSolverTorchSingleBackTrack(d, random_policy, max_score_normalised, nni_tol=0.02)
    mcts_random.solve_timed(iterations)
    print(mcts_random.max_depth)


    rand_nni1 = RandomNni(d, parallel=False)
    rand_nni1.solve_timed(mcts_random.n_nodes)
    print("rand parallel", mcts_random.n_nodes, rand_nni1.time, rand_nni1.obj_val)


    # mcts_t = UtcSolverTorch(d, mixed_policy, average_score_normalised)
    # mcts_t.solve_timed(iterations)
    #

    #
    # fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False, logs=False)
    # fast.solve()
    # print(swa.obj_val, swa_nni.obj_val, mcts_random.obj_val, fast.obj_val)
    # print(swa.time, swa_nni.time, mcts_random.time, fast.method, '\n')
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


