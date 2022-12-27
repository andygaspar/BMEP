import random
import time

import numpy as np
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch

from Old.utc_solver import UtcSolver
from Solvers.SWA.swa_solver import SwaSolver
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch
from Solvers.UCTSolver.utils.utc_utils import nni_landscape

distances = DistanceData()
distances.print_dataset_names()


data_set = distances.get_dataset(3)

d = data_set.get_random_mat(20)

swa_torch = SwaSolverTorch(d)
swa_torch.solve()

adj_mat = swa_torch.solution

mats = nni_landscape(adj_mat, swa_torch.n_taxa, swa_torch.m)







