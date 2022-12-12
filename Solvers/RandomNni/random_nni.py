import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.Random.random_solver import RandomSolver
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch
from Solvers.UCTSolver.utils.utils_rollout import random_policy
from Solvers.solver import Solver


class RandomNni(Solver):
    def __init__(self, d, parallel=False):
        super().__init__(d)
        self.fast_me = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True,
                        triangular_inequality=False, logs=False)
        self.random_solver = RandomSolver(d)
        self.parallel = parallel

    def solve(self, iterations):
        return self.solve_sequential(iterations) if not self.parallel else self.solve_parallel(iterations)

    def solve_sequential(self, iterations):
        best_val, best_sol = 10**5, None
        for i in range(iterations):
            self.random_solver.solve()
            self.fast_me.update_topology(self.random_solver.T)
            self.fast_me.solve()
            if self.fast_me.obj_val < best_val:
                best_val, best_sol= self.fast_me.obj_val, self.fast_me.solution

        self.solution = best_sol
        self.obj_val = best_val

    def solve_parallel(self, iterations):
        d = torch.tensor(self.d, device=self.device)
        adj_mats = self.initial_adj_mat(self.device, iterations)
        obj_vals, adj_mats = random_policy(3, d, adj_mats, self.n_taxa)
        print("here")
        improved, best_val, current_adj = \
            run_nni_search_batch(adj_mats, obj_vals[0], d, self.n_taxa, self.m, self.device)
        idx = torch.argmin(best_val)
        self.obj_val = best_val[idx].item()
        self.solution = current_adj[idx].to('cpu').numpy()
        self.T = self.get_tau(self.solution)
        self.obj_val = self.compute_obj()