import numpy as np
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.Random.random_solver import RandomSolver
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch
from Solvers.UCTSolver.utils.utils_rollout import random_policy
from Solvers.solver import Solver


class RandomNni(Solver):
    def __init__(self, d, parallel=False, spr=True):
        super().__init__(d)
        self.fast_me = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=spr,
                        triangular_inequality=False, logs=False)
        self.random_solver = RandomSolver(d)
        self.parallel = parallel
        self.counter = 0
        self.best_iteration = None
        self.better_solutions = []

    def solve(self, iterations):
        return self.solve_sequential(iterations) if not self.parallel else self.solve_parallel(iterations)

    def solve_sequential(self, iterations, ob_init_val=10, sol_init= None, count_solutions=False):
        best_val, best_sol = 10**5, sol_init
        self.better_solutions.append(sol_init)
        for i in range(iterations):
            self.random_solver.solve()
            # improved, best_val, best_sol = \
            #     run_nni_search(torch.tensor(self.random_solver.solution, device=self.device), self.random_solver.obj_val,
            #                          torch.tensor(self.d, device=self.device), self.n_taxa, self.m, self.powers, self.device)

            self.fast_me.update_topology(self.random_solver.T)
            self.fast_me.solve()
            if count_solutions:
                if self.fast_me.obj_val < ob_init_val:
                    new = True
                    for sol in self.better_solutions:
                        new = not np.array_equal(self.fast_me.solution, sol)
                        if not new:
                            break
                    if new:
                        self.counter += 1
                        print(self.counter)
                        self.better_solutions.append(self.fast_me.solution)
            if self.fast_me.obj_val < best_val:
                best_val, best_sol= self.fast_me.obj_val, self.fast_me.solution
                self.best_iteration = i


        self.solution = best_sol
        self.obj_val = best_val
        self.T = self.get_tau(self.solution)

    def solve_parallel(self, iterations):
        d = torch.tensor(self.d, device=self.device)
        adj_mats = self.initial_adj_mat(self.device, iterations)
        obj_vals, adj_mats = random_policy(3, d, adj_mats, self.n_taxa, self.powers, self.device)
        improved, best_val, current_adj = \
            run_nni_search_batch(adj_mats, obj_vals, d, self.n_taxa, self.m, self.powers, self.device)
        idx = torch.argmin(best_val)
        self.obj_val = best_val[idx].item()
        self.solution = current_adj[idx].to('cpu').numpy()
        self.T = self.get_tau(self.solution)
        self.obj_val = self.compute_obj()