import numpy as np
from FastME.fast_me import FastMeSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.utc_solver import UtcSolver

import sys

from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch

from Data_.data_loader import DistanceData

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)


dim = 20

runs = 6

results = np.zeros((runs, 4))


for run in range(runs):
    print(run)
    d = data_set.generate_random(dim)

    mcts_rand = UtcSolver(d, SwaSolverTorch)
    mcts_rand.solve_timed(20)

    swa_new = SwaSolver(d)
    swa_new.solve_timed()

    # nj_i = NjIlp(d)
    # nj_i.solve()

    mcts = UtcSolver(d)
    mcts.solve_timed(10)

    mcts_1 = UtcSolver(d)
    mcts_1.solve_timed(20)



    mcts_rand1 = UtcSolverTorch(d)
    mcts_rand1.solve_timed(30)

    print(swa_new.time, mcts.time, mcts_1.time, mcts_rand.time, mcts_rand1.time)
    print(swa_new.obj_val, mcts.obj_val, mcts_1.obj_val, mcts_rand.obj_val, mcts_rand1.obj_val)
    print('')

    fast = FastMeSolver(d, bme=False, nni=False, triangular_inequality=False, logs=False)
    fast.solve_timed()

    fast1 = FastMeSolver(d, bme=True, nni=False, triangular_inequality=False, logs=False)
    fast1.solve_timed()

    fast2 = FastMeSolver(d, bme=True, nni=True, triangular_inequality=False, logs=False)
    fast2.solve_timed()

    fast3 = FastMeSolver(d, bme=True, nni=True, digits=17, triangular_inequality=True, logs=False)
    fast3.solve_timed()

    fast4 = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True, triangular_inequality=True, logs=False)
    fast4.solve_timed()

    print(fast.time, fast1.time, fast2.time, fast3.time, fast4.time)
    print(fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, fast4.obj_val, "\n\n")

