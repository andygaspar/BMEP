import random

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

random.seed(0)
np.random.seed(0)
iterations = 1


for dim in [20, 30, 40]:
    for it in [1, 3, 5, 10]:
        print(dim, it)
        d = data_set.get_random_mat(dim)

        guided_rand = GuidedRandSolver(d, it)
        guided_rand.solve_timed()
        print(guided_rand.obj_val, guided_rand.time)
        print(guided_rand.n_nodes, 'nodes', guided_rand.max_depth, ' depth', guided_rand.n_trajectories, 'tj',
              guided_rand.n_trees, 'trees')


'''
from ast import literal_eval

my_str = '(1,2,3,4)'

# ✅ convert string representation of tuple to tuple (ast.literal_eval())

my_tuple = literal_eval(my_str)
print(my_tuple)  # 👉️ (1, 2, 3, 4)
print(type(my_tuple))  # 👉️ <class 'tuple'>

# ---------------------------------

# ✅ convert string representation of tuple to tuple (using split())

my_tuple = tuple(map(int, my_str.strip('()').split(',')))
print(my_tuple)  # 👉️ (1, 2, 3, 4)

# ---------------------------------

# 👇️ for comma-separated string

my_str = "1,2,3,4"

my_tuple = tuple(int(digit) for digit in my_str.split(","))
print(my_tuple)  # 👉️ (1, 2, 3, 4)


'''