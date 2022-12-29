import random

import torch

from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch
from Solvers.solver import Solver


def swa_policy(start, d, adj_mats: torch.tensor, n_taxa, powers, device):
    batch_size = adj_mats.shape[0]
    obj_vals = None
    for step in range(start, n_taxa):
        idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
        idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(device), idxs_list[1], idxs_list[2])
        minor_idxs = torch.tensor([j for j in range(step + 1)]
                                  + [j for j in range(n_taxa, n_taxa + step - 1)]).to(device)

        repetitions = 3 + (step - 3) * 2

        adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

        sol = Solver.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=n_taxa)
        obj_vals = Solver.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                              d[:step + 1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                              n_taxa=step + 1, powers=powers, device=device)
        obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
        adj_mats = sol.unsqueeze(0).view(batch_size, repetitions, adj_mats.shape[1],
                                         adj_mats.shape[2])[range(batch_size), obj_vals.indices, :, :]

    return obj_vals.values, adj_mats


def random_policy(start, d, adj_mats, n_taxa,  powers, device, iteration=None):
    batch_size = adj_mats.shape[0]
    for step in range(start, n_taxa):
        choices = 3 + (step - 3) * 2
        idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch_size, -1, 3)
        rand_idxs = random.choices(range(choices), k=batch_size)
        idxs_list = idxs_list[[range(batch_size)], rand_idxs, :]
        idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
        adj_mats = Solver.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=n_taxa)

    obj_vals = Solver.compute_obj_val_batch(adj_mats, d, powers, n_taxa, device)
    return obj_vals, adj_mats


def mixed_policy(node, start, adj_mats, iteration):
    if iteration < 30:
        return swa_policy(node, start, adj_mats)
    else:
        return random_policy(node, start, adj_mats)


def swa_nni_policy(node, start=3, adj_mats: torch.tensor = None, every=5, after=10, n_iter=5, n_final_iter=10):
    batch_size = adj_mats.shape[0]
    obj_vals = None
    n_internals = node._n_taxa * 2 -2 - node._n_taxa    # to optimise
    for step in range(start, node._n_taxa):
        idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
        idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(node._device), idxs_list[1], idxs_list[2])
        minor_idxs = torch.tensor([j for j in range(step + 1)]
                                  + [j for j in range(node._n_taxa, node._n_taxa + step - 1)]).to(node._device)

        repetitions = 3 + (step - 3) * 2

        adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

        sol = node.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=node._n_taxa)
        obj_vals = node.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                              node._d[:step + 1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                              step + 1)
        obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
        adj_mats = sol.unsqueeze(0).view(batch_size, repetitions, adj_mats.shape[1],
                                         adj_mats.shape[2])[range(batch_size), obj_vals.indices, :, :]

        # if step > after and (step - 3) % every == 0:
        if step == node._n_taxa - 1 or (step > after and (step - 3) % every == 0):
            n_iter = n_iter if step < node._n_taxa - 1 else n_final_iter
            improved, best_val, best_solution = \
                run_nni_search_batch(n_iter, adj_mats, obj_vals.values, node._d, node._n_taxa, node._m, node._device)
            if torch.any(improved):
                adj_mats[improved] = adjust_matrices(best_solution[improved], step, n_internals, node._n_taxa)

    return obj_vals.values, adj_mats


def adjust_matrices(adj_mat, last_inserted_taxa, n_internals, n_taxa):
    adj_mat = adj_mat.unsqueeze(1).repeat(1, 2, 1, 1)

    # reorder matrix according to Pardi
    for step in range(last_inserted_taxa + n_internals, n_taxa, -1):
        not_in_position = torch.argwhere(adj_mat[:, 1, step, last_inserted_taxa] == 0).squeeze(1)
        if len(not_in_position)>0:
            a = adj_mat[not_in_position, 1, last_inserted_taxa, :]
            idxs = torch.argwhere(adj_mat[not_in_position, 1, last_inserted_taxa, :] == 1)
            adj_mat = permute(adj_mat, step, not_in_position, idxs)

        adj_mat[:, 1, :, last_inserted_taxa] = adj_mat[:, 1, last_inserted_taxa, :] = 0

        i = torch.nonzero(adj_mat[:, 1, step, :])
        idx = i[:, 1]
        idxs = idx.reshape(-1, 2).T
        batch_idxs = range(adj_mat.shape[0])
        adj_mat[batch_idxs, 1, idxs[0], idxs[1]] = adj_mat[batch_idxs, 1, idxs[1], idxs[0]] = 1
        adj_mat[:, 1, :, step] = adj_mat[:, 1, step, :] = 0

        last_inserted_taxa -= 1
    adj_mat = adj_mat[:, 0, :, :]
    return adj_mat


def permute(adj_mats, step, not_in_position, idx):
    adj_mats[not_in_position, :, step, :] += adj_mats[not_in_position, :, idx[:, 1], :]
    adj_mats[not_in_position, :, idx[:, 1], :] = adj_mats[not_in_position, :, step, :] - adj_mats[not_in_position, :, idx[:, 1], :]
    adj_mats[not_in_position, :, step, :] -= adj_mats[not_in_position, :, idx[:, 1], :]

    adj_mats[not_in_position, :, :, step] += adj_mats[not_in_position, :, :, idx[:, 1]]
    adj_mats[not_in_position, :, :, idx[:, 1]] = adj_mats[not_in_position, :, :, step] - adj_mats[not_in_position, :, :, idx[:, 1]]
    adj_mats[not_in_position, :, :, step] -= adj_mats[not_in_position, :, :, idx[:, 1]]
    return adj_mats