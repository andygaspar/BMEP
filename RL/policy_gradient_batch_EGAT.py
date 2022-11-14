import copy
import multiprocessing as mp
import torch.multiprocessing

import networkx as nx
import numpy as np
import torch.nn.init

from RL.Batches.batch import Batch
from RL.Batches.batch_EGAT import BatchEGAT
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.solver import Solver

def sort_d(d):
    dist_sum = np.sum(d, axis=0)
    order = np.argsort(dist_sum)[::-1]
    sorted_d = np.zeros_like(d)
    for i in order:
        for j in order:
            sorted_d[i, j] = d[order[i], order[j]]
    return sorted_d


def make_d_batch(d):
    m = d.shape[0] * 2 - 2
    d = sort_d(d)
    d_with_internals = np.zeros((m, m))
    d_with_internals[: d.shape[0], : d.shape[0]] = d
    return d_with_internals


def get_tau_tensor_(adj_mat):
    g = nx.from_numpy_matrix(adj_mat)
    tau = nx.floyd_warshall_numpy(g)
    tau[np.isinf(tau)] = 0
    tau = torch.tensor(tau).to(torch.float)
    return tau


def compute_obj_and_baseline_(data):
    adj_mat, d = data
    swa = SwaSolver(d)
    n_taxa = swa.n_taxa
    swa.solve()
    baseline = swa.obj_val
    g = nx.from_numpy_matrix(adj_mat)
    Tau = nx.floyd_warshall_numpy(g)[:n_taxa, :n_taxa]
    return np.sum([d[i, j] / 2 ** (Tau[i, j]) for i in range(n_taxa) for j in range(n_taxa)]), baseline


class PolicyGradientEGAT(Solver):

    def __init__(self, net):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_procs = mp.cpu_count()
        self.batch_size = None
        super().__init__(None)
        self.net = net

        self.adj_mats = []
        self.better, self.equal, self.mean_difference = None, None, None

    def episode(self, d_list, n_taxa, replay_memory: BatchEGAT):

        with torch.no_grad():
            n_problems = len(d_list)
            self.n_taxa = n_taxa
            self.m = self.n_taxa * 2 - 2
            pool = mp.Pool(self.num_procs)
            results = pool.map(make_d_batch, d_list)
            pool.close()
            pool.join()
            d = np.array(results)
            d = torch.tensor(d).to(torch.float).to(self.device)

            adj_mat = self.initial_adj_mat(device=self.device, n_problems=n_problems)
            tau, idxs = None, None
            variance_probs = []

            for i in range(3, self.n_taxa):
                tau = self.get_taus(adj_mat, tau)
                state = self.net.get_net_input(adj_mat, d, tau, self.m, self.n_taxa, step=i, n_problems=n_problems)
                probs, _ = self.net(state)
                # print(probs[0])
                variance_probs.append(torch.var(probs[probs > 0.001]).item())
                prob_dist = torch.distributions.Categorical(probs)  # probs should be of size batch x classes
                action = prob_dist.sample()
                idxs = (torch.arange(0, n_problems, dtype=torch.long), torch.div(action, self.m, rounding_mode='trunc'),
                        action % self.m)
                taxa_embeddings, internal_embeddings, message_embeddings, problem_mask, active_nodes_mask, action_mask \
                    = state
                replay_memory.add_states(taxa_embeddings, internal_embeddings, message_embeddings,
                                         problem_mask, active_nodes_mask, action_mask, action, self.m)
                adj_mat = self.add_nodes(adj_mat, idxs, new_node_idx=i, n=self.n_taxa)

            adj_mats_np = adj_mat.to('cpu').numpy()
            pool = mp.Pool(self.num_procs)
            results = pool.map(compute_obj_and_baseline_,
                               [(adj_mats_np[i], d_list[i]) for i in range(adj_mat.shape[0])])
            results = torch.tensor(results, dtype=torch.float).to(self.device)
            pool.close()
            pool.join()
            obj_vals = results[:, 0]
            baseline = results[:, 1]
            replay_memory.add_rewards_baselines(obj_vals, baseline, self.n_taxa)
            better = sum(obj_vals < baseline).item()
            equal = sum(obj_vals == baseline).item()

            return torch.mean((obj_vals - baseline) / baseline).item(), better, equal, variance_probs

    def standings(self):
        return self.mean_difference, self.better, self.equal

    def get_taus(self, adj_mat, taus):
        n = self.n_taxa
        if taus is None:
            taus = torch.zeros_like(adj_mat)
            taus[:, 0, 1] = taus[:, 1, 0] = taus[:, 0, 2] = taus[:, 2, 0] = taus[:, 1, 2] = taus[:, 2, 1] = 2
            taus[:, 0, n] = taus[:, n, 0] = taus[:, n, 2] = taus[:, 2, n] = taus[:, 1, n] = taus[:, n, 1] = 1
        else:
            adj_mats_np = adj_mat.to('cpu').numpy()
            pool = mp.Pool(self.num_procs)
            taus = pool.map(get_tau_tensor_, [adj_mats_np[i] for i in range(adj_mat.shape[0])])
            pool.close()
            pool.join()
            taus = torch.stack(taus).to(torch.float).to(self.device)

        return taus

    @staticmethod
    def add_nodes(adj_mat, idxs: torch.tensor, new_node_idx, n):
        adj_mat[idxs] = adj_mat[idxs[0], idxs[2], idxs[1]] = 0  # detach selected
        adj_mat[idxs[0], idxs[1], n + new_node_idx - 2] = adj_mat[
            idxs[0], n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[idxs[0], idxs[2], n + new_node_idx - 2] = adj_mat[
            idxs[0], n + new_node_idx - 2, idxs[2]] = 1  # reattach selected to new
        adj_mat[idxs[0], new_node_idx, n + new_node_idx - 2] = adj_mat[
            idxs[0], n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat


