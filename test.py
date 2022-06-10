from Old.Test_matrices.test_mats import matrix_dict
from Pardi.pardi import Pardi


# matrix dict is imported from folder: Old.Test_matrices.test_mats
# it is a dict with few test Tau matrices

for T in matrix_dict.values():
    print("Tau\n", T)
    pardi = Pardi(T)
    pardi.get_pardi_assignment(True)
    pardi.get_graph(show=True)
    print("\n")
