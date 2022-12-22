import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.FastME.pharser_newik.newwik_handler import get_adj_from_nwk, compute_newick
from Solvers.NJ_ILP.nj_ilp import NjIlp
from Solvers.Random.guided_random import GuidedRandSolver
from Solvers.RandomNni.random_nni import RandomNni
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.SWA.swa_solver_torch_nni import SwaSolverTorchNni
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_ import UtcSolverTorchSingleBackTrack
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorchBackTrack2
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, mixed_policy, swa_nni_policy, random_policy
from Solvers.UCTSolver.utils.utils_scores import average_score_normalised, max_score_normalised
from check_nni import cn

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 30

runs = 2

results = np.zeros((runs, 4))
seed = 0
random.seed(seed)
np.random.seed(seed)
iterations = 1000

idxs_list = []

run = 0



while len(idxs_list) < 500:
    print('\n', run, '***** ',len(idxs_list))
    d, idx = data_set.get_random_mat(dim, return_idxs=True)


    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True,
                        triangular_inequality=False, logs=False)
    fast.solve()

    comparison = RandomNni(d, parallel=False)
    comparison.solve_sequential(iterations, fast.obj_val)

    if comparison.counter >= 30:
        idxs_list.append(idx)
    print(fast.obj_val, comparison.obj_val)
    run += 1


df = pd.DataFrame({'idxs': idxs_list})
df.to_csv('test_instances_idxs.csv', index_label=False, index=False)

print('n taxa', dim)
print('single fastME solution obj', fast.obj_val)
print('1000 random initialisation + fastME best obj', comparison.obj_val)
print('number of solutions better than fastME found', comparison.counter)






