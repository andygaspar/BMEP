import random
import pandas as pd
import numpy as np
from FastME.fast_me import FastMeSolver
from Solvers.NJ_ILP.nj_ilp import NjIlp
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.utc_solver import UtcSolver

import sys

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch_1 import UtcSolverTorch_1

distances = DistanceData()
distances.print_dataset_names()

results, sizes, data_ = [], [], []


random.seed(0)
np.random.seed(0)
iterations = 0

dims = [20, 25, 30, 35]

for key in distances.data_sets.keys():

    data_set = distances.get_dataset(key)
    cases = [size for size in dims if size <= data_set.size]
    if len(cases) < 4:
        cases.append(data_set.size)
    for dim in cases:
        data_.append(key)
        sizes.append(dim)
        d = data_set.get_minor(dim)

        nj_i = NjIlp(d)
        nj_i.solve(60)

        swa = SwaSolverTorch(d)
        swa.solve_timed()

        mcts_t_1 = UtcSolverTorch_1(d)
        mcts_t_1.solve_timed(300)

        fast = FastMeSolver(d, bme=False, nni=False, triangular_inequality=False, logs=False)
        fast.solve_timed()
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
        fast4 = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=True, logs=False)
        fast4.solve_timed()

        results.append([swa.obj_val, nj_i.obj_val, mcts_t_1.obj_val, fast.obj_val, fast4.obj_val])

    # print(fast.time, fast1.time, fast2.time, fast3.time, fast4.time)
    # print(fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, fast4.obj_val, "\n\n")
results = np.array(results)
df = pd.DataFrame({'data_set': data_, 'dim': sizes, 'swa': results[:, 0], 'ni_ip': results[:, 1],
                   'mcts': results[:, 2], 'fast_1': results[:, 3], 'fast_2': results[:, 4]})
df.to_csv("test_results", index_label=False, index=False)

import pandas as pd
df = pd.read_csv("test_results.csv")