import time

from FastME.fast_me import FastMeSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch

from Old.utc_solver import UtcSolver
from Solvers.SWA.swa_solver import SwaSolver
from Data_.data_loader import DistanceData
from Solvers.UCTSolver.utc_solver_torch import UtcSolverTorch

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)

d = data_set.get_minor(6)

swa = SwaSolver(d)
swa.solve_timed()

mcts_torch = UtcSolverTorch(d)
mcts_torch.solve_timed(3)


swa_torch = SwaSolverTorch(d)
print(swa_torch.device)
swa_torch.solve_timed()


print(swa.time, swa_torch.time, mcts_torch.time)
print(swa.obj_val, swa_torch.obj_val, mcts_torch.obj_val)

fast = FastMeSolver(d)
fast.solve()

t = time.time()
mcts_solver = UtcSolver(d)
mcts_solver.solve(1)
print('mcts ', time.time() - t)

p = mcts_solver.T
# fast1 = FastMeSolver(d, init_topology=newick_tree)
# fast1.solve()





