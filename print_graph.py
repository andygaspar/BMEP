import copy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


np.array([[0, 5, 5, 4, 2, 3],
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
n = s.shape[0] + 1
internal = []
removed = []
internal_map = {}
while not over:
    idxs = np.array(np.where(np.triu(s[1:, 1:]) == 2)).T

    if idxs.shape[0] > 1:
        for idx in idxs:
            i, j = idx
            if i not in internal and j not in internal:
                internal.append(i)
                G.add_nodes_from([(i, {"color": "red"})])
                internal.append(j)
                G.add_nodes_from([(j, {"color": "red"})])

                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(i, n + k), (j, n + k)])
                internal_map[i] = n + k
                k += 1

            elif i in internal and j not in internal:
                internal.append(j)

                G.add_nodes_from([(j, {"color": "red"})])
                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(internal_map[i], n + k), (j, n + k)])
                k += 1
            elif j in internal and j not in internal:
                internal.append(j)
                G.add_nodes_from([(j, {"color": "red"})])
                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(internal_map[i], n + k), (j, n + k)])
                k += 1
            else:
                G.add_nodes_from([(n + k, {"color": "green"})])
                G.add_edges_from([(internal_map[i], n + k), (internal_map[j], n + k)])
                k += 1

            s[i] -= 1
            s[:, i] -= 1
            s[j] += n
            s[:, j] += n

            nx.draw(G, node_color=[G.nodes[node]["color"] for node in G.nodes], with_labels=True, font_weight='bold')
            plt.show()
    elif idxs.shape[0] == 0:
        over = True

    else:
        over = True

nx.draw(G, node_color= [G.nodes[node]["color"] for node in G.nodes],with_labels=True, font_weight='bold')
plt.show()