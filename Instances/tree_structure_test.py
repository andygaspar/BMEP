import numpy as np
from itertools import combinations, permutations


def kraft(A):
    return np.prod([sum([2. ** (-A[i, j]) for j in range(6) if i != j]) == 0.5 for i in range(6)], dtype=bool)


def manifold(A):
    return sum([A[i, j] * 2. ** (-A[i, j]) for i in range(6) for j in range(6) if i != j]) == 2 * A.shape[0] - 3


def symmetry(A):
    return np.prod([A[i, j] == A[j, i] for i in range(A.shape[0]) for j in range(A.shape[0])], dtype=bool)


def buneman(A):
    bun = []
    for comb in list(combinations(range(A.shape[0]), 4)):
        for perm in list(permutations(comb)):
            i, j, q, p = perm
            a = A[i, j] + A[p, q] + 2 <= A[i, q] + A[j, p] == A[i, p] + A[j, q]
            b = A[i, q] + A[j, p] + 2 <= A[i, j] + A[p, q] == A[i, p] + A[j, q]
            c = A[i, p] + A[j, q] + 2 <= A[i, j] + A[p, q] == A[i, q] + A[j, p]
            bun.append(True if sum([a, b, c]) == 1 else False)

    return np.prod(bun, dtype=bool)


def buneman_2(A):
    bun = []
    for comb in list(combinations(range(A.shape[0]), 4)):
        for perm in list(permutations(comb)):
            i, j, q, t = perm
            y = A[i, t] + A[j, q] >= A[i, q] + A[j, t]
            a = A[i, j] + A[q, t] <= A[i, q] + A[j, t] + (2 * A.shape[0] - 2) * y
            b = A[i, j] + A[q, t] <= A[i, t] + A[j, q] + (2 * A.shape[0] - 2) * (1 - y)
            c = A[i, j] + A[q, t] >= A[i, t] + A[j, q] - (2 * A.shape[0] - 2) * y
            bun.append(True if sum([a, b, c]) == 3 else False)

    return np.prod(bun, dtype=bool)


def compare_y_solution(y_sol, sol):
    comparison = []
    for comb in list(combinations(range(sol.shape[0]), 4)):
        for perm in list(permutations(comb)):
            i, j, q, t = perm
            yy = sol[i, t] + sol[j, q] >= sol[i, q] + sol[j, t]
            comparison.append(yy == y_sol[i, j, q, t])

    return np.prod(comparison, dtype=bool)


def run_test(A):
    print(symmetry(A), kraft(A), manifold(A), buneman(A), buneman_2(A))





