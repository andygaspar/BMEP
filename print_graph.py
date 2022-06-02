import copy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


z = np.array([[0, 5, 5, 4, 2, 3],
          [5, 0, 2, 3, 5, 4],
          [5, 2, 0, 3, 5, 4],
          [4, 3, 3, 0, 4, 3],
          [2, 5, 5, 4, 0, 3],
          [3, 4, 4, 3, 3, 0]])

solution_xp = np.array([[0, 2, 4, 5, 5, 3],
                        [2, 0, 4, 5, 5, 3],
                        [4, 4, 0, 3, 3, 3],
                        [5, 5, 3, 0, 2, 4],
                        [5, 5, 3, 2, 0, 4],
                        [3, 3, 3, 4, 4, 0]])

G = nx.Graph()
s = copy.deepcopy(solution_xp)
s = np.vstack([range(s.shape[0]), s])
col = np.array([0]+[i for i in range(s.shape[0]-1)])
s = np.c_[col, s]
over = False
k = 0
n = s.shape[0] - 1
internal = []
removed = []
internal_map = {}
while not over:
    print(s, '\n')

    if s.shape[0] > 3:
        idx = np.array(np.where(np.triu(s[1:, 1:]) == 2)).T[0] + 1
        i, j = idx
        if s[0, i] < n and s[0, j] < n:
            internal.append(i - 1)
            G.add_nodes_from([(s[0, i], {"color": "red"})])
            G.add_nodes_from([(s[0, j], {"color": "red"})])

            G.add_nodes_from([(n + k, {"color": "green"})])
            G.add_edges_from([(s[0, i], n + k), (s[0, j], n + k)])

        elif s[0, j] < n:
            # k += 1
            G.add_nodes_from([(s[0, j], {"color": "red"})])
            G.add_nodes_from([(n + k, {"color": "green"})])
            G.add_edges_from([(s[0, i], n + k), (s[0, j], n + k)])

        else:
            # k += 1
            G.add_nodes_from([(n + k, {"color": "green"})])
            G.add_edges_from([(s[0, i], n + k), (s[0, j], n + k)])

        i, j = idx
        s[i, 0] = n + k
        s[0, i] = n + k
        s[i, 1:] -= 1
        s[1:, i] -= 1
        k += 1
        s = np.delete(s, j, axis=0)
        s = np.delete(s, j, axis=1)
    else:
        idx = np.array(np.where(np.triu(s[1:, 1:]) == 1)).T[0] + 1
        i, j = idx
        k += 1
        G.add_nodes_from([(s[0, j], {"color": "red"})])
        G.add_edges_from([(s[0, i], s[0, j])])
        over = True
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()


nx.draw(G, node_color= [G.nodes[node]["color"] for node in G.nodes],with_labels=True, font_weight='bold')
plt.show()