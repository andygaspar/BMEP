from Solvers.FastME.fast_me import FastMeSolver
from Solvers.Random.random_solver import RandomSolver
from Solvers.solver import Solver


class RandomNni(Solver):
    def __init__(self, d):
        super().__init__(d)
        self.fast_me = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True,
                        triangular_inequality=False, logs=False)
        self.random_solver = RandomSolver(d)

    def solve(self, iterations):
        best_val, best_sol = 10**5, None
        for i in range(iterations):
            self.random_solver.solve()
            self.fast_me.update_topology(self.random_solver.T)
            self.fast_me.solve()
            if self.fast_me.obj_val < best_val:
                best_val, best_sol= self.fast_me.obj_val, self.fast_me.solution

        self.solution = best_sol
        self.obj_val = best_val