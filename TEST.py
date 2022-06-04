import numpy as np
import string

T = np.array([[0, 4, 2, 5, 5, 4, 4],
              [4, 0, 4, 5, 5, 2, 4],
              [2, 4, 0, 5, 5, 4, 4],
              [5, 5, 5, 0, 2, 5, 3],
              [5, 5, 5, 2, 0, 5, 3],
              [4, 2, 4, 5, 5, 0, 4],
              [4, 4, 4, 3, 3, 4, 0]])


def step(leaf, T, labels):
    assignment = []
    match = np.where(T[leaf] == 2)[0]
    if match.shape[0] > 0:
        match = match[0]
        if match < leaf:
            chery = match
            assignment.append(labels[leaf] + "->" + labels[chery])
            T[:, chery] -= - 1
            T[chery] -= 1
            T[chery, chery] += 1
            T[:, leaf] = - 1
            T[leaf] = 1
        else:
            pass
    else:
        assignment.append(labels[leaf] + "->" + str(leaf - 2))
        T[:, -1] -= 1

    print(assignment)
    return T


labels = [s for s in string.ascii_uppercase[:T.shape[0]]]

T = step(T, labels)
T = step(T, labels)
T = step(T, labels)
T = step(T, labels)

print(T)