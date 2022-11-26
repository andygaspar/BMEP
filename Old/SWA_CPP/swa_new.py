import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer
from Solvers.solver import Solver
import scipy


class SwaSolverNew(Solver):

    def __init__(self, d, sorted_d=False):
        super(SwaSolverNew, self).__init__(d, sorted_d)
        self.obj_val_ = None
        ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64,
                                              ndim=2,
                                              flags="C")
        self.lib = ctypes.CDLL('Solvers/SWA/bridge_swa.so')
        self.lib.print_.argtypes = [ctypes.POINTER(ctypes.c_double),  ctypes.c_short]
        self.lib.swa_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_short]
        self.lib.swa_.restype = ctypes.c_double

    def solve(self, start=3, adj_mat=None):
        self.obj_val = self.lib.swa_(self.d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_short(self.n_taxa))
        print(self.obj_val)
        self.lib.print_(self.d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_short(self.n_taxa))
        print(self.obj_val, "llll")



