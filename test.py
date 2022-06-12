import numpy as np

from Old.Test_matrices.test_mats import matrix_dict
from Pardi.pardi import Pardi

# matrix dict is imported from folder: Old.Test_matrices.test_mats
# it is a dict with few test Tau matrices

for T in list(matrix_dict.values())[-1:]:
    print("Tau\n", T)
    pardi = Pardi(T)
    pardi.get_pardi_assignment(True)
    pardi.get_graph(show=True)
    for m in pardi.adj_mats:
        print("\n")
        print(m)
        # linalg.multi_dot



b = np.array([range(1, 9), range(1, 9)])

A = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

