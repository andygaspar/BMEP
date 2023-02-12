import random

import numpy as np
import pandas as pd
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.Random.guided_random import GuidedRandSolver
from Solvers.RandomNni.random_nni import RandomNni
from Solvers.RandomNni.phyloga import PhyloGA
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchSingleBackTrack
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorchBackTrack2
from Solvers.UCTSolver.utils.utils_rollout import random_policy
from Solvers.UCTSolver.utils.utils_scores import max_score_normalised, max_score_normalised_dict, \
    average_score_normalised_dict

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 70

runs = 20


# random.seed(0)
# np.random.seed(0)
max_iterations = 100

results = []
k = 0

batch = 15
for dim in [40]:#, 50, 60, 70, 80, 90]:
    print('*************** ', dim)
    for run in range(runs):
        results.append([])
        results[k] += [dim, run]
        print('\n', run)
        d = data_set.get_random_mat(dim)
        rand_tj = PhyloGA(d, batch=batch, max_iterations=100)
        rand_tj.solve_timed()
        print(rand_tj.time, "rand", rand_tj.obj_val)
        rand_tj_tj = tuple(rand_tj.tree_climb(torch.tensor(rand_tj.solution).unsqueeze(0)).to('cpu').tolist()[0])
        print(rand_tj_tj)

        rand_tj1 = PhyloGA(d, batch=30, max_iterations=100)
        rand_tj1.solve_timed()
        print(rand_tj1.time, "rand", rand_tj1.obj_val)
        rand_tj1_tj = tuple(rand_tj1.tree_climb(torch.tensor(rand_tj1.solution).unsqueeze(0)).to('cpu').tolist()[0])
        print(rand_tj1_tj)

        rand_nni1 = RandomNni(d, parallel=False, spr=True)
        rand_nni1.solve_timed(rand_tj.iterations)
        print("rand parallel", rand_nni1.time, rand_nni1.obj_val)
        rand_nni_tj = tuple(rand_tj.tree_climb(torch.tensor(rand_nni1.solution).unsqueeze(0)).to('cpu').tolist()[0])
        print(rand_nni_tj)

        #
        fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=False, logs=False)
        fast.solve_all_flags()
        print('fast', fast.obj_val)
        fast_tj = tuple(rand_tj.tree_climb(torch.tensor(fast.solution).unsqueeze(0)).to('cpu').tolist()[0])
        print(fast_tj)
        results[k] += [fast.obj_val, rand_nni1.obj_val,  rand_tj.obj_val, rand_tj1.obj_val, fast.time, rand_nni1.time,
                       rand_tj.time, rand_tj1.time, fast.method]
        results[k] += [fast_tj, rand_nni_tj, rand_tj_tj, rand_tj1_tj]
        k += 1


res = np.matrix(results)
df = pd.DataFrame(np.mat(results), columns=['Taxa', 'Run', 'fast_obj', 'random_obj', 'td_fast15_obj', 'td_fast30_obj', 'fast_time',
                                            'rand_time', 'td_fast15_time', 'td_fast30_time', 'fastMe_init', 'fast_traj', 'rand_traj',
                                            'td_fast15_tj', 'td_fast30_tj'])
print(df)
# df["fast_improvement"] = df['fast_obj']/df['td_fast_obj'] - 1
# df['random_improvement'] = df['random_obj']/df['td_fast_obj'] - 1
# df.to_csv('test_td_fast2.csv', index_label=False, index=False)
# results = np.array(results)
# df = pd.DataFrame({"mcts": results[:, 0], "lp_nj": results[:, 1], "fast": results[:, 2]})
# df.to_csv("test_new.csv", index_label=False, index=False)


import pandas as pd

df = pd.read_csv("test_td_fast.csv")




