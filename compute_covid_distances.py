import time
import numpy as np
import pandas

def euclidian(x, y):
    return np.sum((x - y) ** 2)


def KL(x, y):
    return np.sum(x * np.log2((x + 1) / (y + 1))), np.sum(y * np.log2((y + 1) / (x + 1)))


def cos_evol(x, y):
    cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    evol = -np.log((1 + cos) / 2)
    return cos, evol

c14 = np.loadtxt('covid14.txt')
c14_dist_euclidian = np.zeros((c14.shape[0], c14.shape[0]))
t = time.time()
for i in range(c14.shape[0]):
    for j in range(i + 1, c14.shape[0]):
        c14_dist_euclidian[i, j] = euclidian(c14[0], c14[1])

print(time.time() - t)
c14[0]

c14[0].sum()
min(np.triu(c14_dist_euclidian).flatten())

np.unique(c14[0])

