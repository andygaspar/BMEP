import time

import torch

from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.solver import Solver


def nni_landscape_batch(adj_mat, n_taxa, mat_size):
    batch_size = adj_mat.shape[0]
    mask = torch.triu(torch.ones_like(adj_mat))
    max_taxa = int((mat_size + 2) / 2)
    mask[:, :max_taxa, :] = 0
    internal_branches = torch.nonzero((adj_mat * mask))
    step = len(internal_branches) // batch_size
    internal_branches = torch.hstack([internal_branches, internal_branches[:, torch.tensor([0, 2, 1])]]).view((-1, 3))
    internal_branches[:, 0] = torch.arange(0, len(internal_branches)) // 2

    nni_switches = torch.repeat_interleave(adj_mat, step, dim=0)
    nni_switches[(internal_branches[:, 0], internal_branches[:, 1], internal_branches[:, 2])] = 0

    nni_switches = torch.nonzero(nni_switches[(internal_branches[:, 0], internal_branches[:, 1])])
    nni_switches[:, 0] = torch.repeat_interleave(internal_branches[:, 1], repeats=2)
    nni_switches = nni_switches.reshape((-1, 8))
    nni_switches[:, 3] = nni_switches[:, 1]
    to_remove = nni_switches[:, [0, 1, 4, 5, 2, 3, 6, 7]].reshape((-1, 2))
    nni_switches = nni_switches[:, [0, 5, 4, 1, 2, 7, 6, 3]].reshape((-1, 2))
    idxs = torch.arange(len(nni_switches)) // 2

    new_trees = torch.repeat_interleave(adj_mat, 2 * step, dim=0)
    new_trees[(idxs, nni_switches[:, 0], nni_switches[:, 1])] = 1
    new_trees[(idxs, nni_switches[:, 1], nni_switches[:, 0])] = 1
    new_trees[(idxs, to_remove[:, 0], to_remove[:, 1])] = 0
    new_trees[(idxs, to_remove[:, 1], to_remove[:, 0])] = 0

    return new_trees.view((batch_size, 2*step, mat_size, mat_size))


def run_nni_search_batch(current_adj, best_val, d, n_taxa, m, powers, device):
    sol = current_adj
    batch_size =  size = current_adj.shape[0]
    improved = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
    idxs = range(batch_size)
    all_idxs = torch.tensor([idxs, idxs], device=device).T
    iterations = 0

    comp_total = 0

    while sol.shape[0] > 0:
        improved[:] = False
        expl_trees = nni_landscape_batch(sol, n_taxa, m)
        batch_size = expl_trees.shape[0]
        obj_vals= Solver.compute_obj_val_batch(expl_trees.view(batch_size*expl_trees.shape[1], m , m), d, powers, n_taxa, device)

        new_obj_val = torch.min(obj_vals.view(batch_size, -1), dim=-1)
        sol = expl_trees[range(batch_size), new_obj_val[1]]
        idxs = torch.argwhere(best_val[all_idxs[:, 1]] > new_obj_val.values).squeeze(1)
        current_adj[all_idxs[idxs, 1]] =  sol[idxs]
        best_val[all_idxs[idxs][:, 1]] = new_obj_val.values[idxs]
        if idxs.shape[0] < size:
            # print(idxs.shape[0], iterations)
            size = idxs.shape[0]
        sol = sol[idxs]
        all_idxs = torch.tensor([range(idxs.shape[0]), all_idxs[idxs, 1]], device=device).T
        iterations += 1
    print(iterations, "iterations", comp_total)
    return improved, best_val, current_adj


def run_nni_search_batch_for_tracking(current_adj, best_values, d, n_taxa, m, powers, device):
    sol = current_adj
    best_val, obj_vals = torch.clone(best_values), None
    batch_size =  size = current_adj.shape[0]
    improved = torch.ones(size=(batch_size,), dtype=torch.bool, device=device)
    idxs = range(batch_size)
    all_idxs = torch.tensor([idxs, idxs], device=device).T
    iterations = 0

    trees, objs = [], []

    while sol.shape[0] > 0:
        improved[:] = False
        expl_trees = nni_landscape_batch(sol, n_taxa, m)
        batch_size = expl_trees.shape[0]
        obj_vals= Solver.compute_obj_val_batch(expl_trees.view(batch_size*expl_trees.shape[1], m , m), d, powers, n_taxa, device)

        trees.append(expl_trees.view(batch_size*expl_trees.shape[1], m , m))
        objs.append(obj_vals.view(batch_size*expl_trees.shape[1]))

        new_obj_val = torch.min(obj_vals.view(batch_size, -1), dim=-1)
        sol = expl_trees[range(batch_size), new_obj_val[1]]
        idxs = torch.argwhere(best_val[all_idxs[:, 1]] > new_obj_val.values).squeeze(1)
        current_adj[all_idxs[idxs, 1]] =  sol[idxs]
        best_val[all_idxs[idxs][:, 1]] = new_obj_val.values[idxs]
        if idxs.shape[0] < size:
            # print(idxs.shape[0], iterations)
            size = idxs.shape[0]
        sol = sol[idxs]
        all_idxs = torch.tensor([range(idxs.shape[0]), all_idxs[idxs, 1]], device=device).T
        iterations += 1
    best_val = torch.min(best_val, dim=-1)
    print(iterations, "iterations", best_val[0].item())

    return best_val[0], current_adj[best_val[1]], trees, objs
