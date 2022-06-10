from Old.Test_matrices.test_mats import matrix_dict
from Pardi.pardi import Pardi


for T in matrix_dict.values():
    pardi = Pardi(T)
    pardi.get_pardi_assignment(True)
    pardi.get_graph(show=True)
