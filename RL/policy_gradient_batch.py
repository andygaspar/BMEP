import copy
import multiprocessing as mp
import numpy as np
import torch.nn.init

from Solvers.solver import Solver


class PolicyGradientBatchEpisode(Solver):

    def __init__(self, net, optim):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_procs = mp.cpu_count()
        self.batch_size = None
        # d_with_internals[:self.n_taxa, :self.n_taxa] = d if type(d) == torch.Tensor else torch.tensor(d)
        # d_list_sorted = [self.]
        # d = torch.tensor(d).to(torch.float).to(self.device)
        super().__init__(None)
        self.net = net

        self.optimiser = optim
        self.adj_mats = []
        self.loss = None

    def make_d_batch(self, d):
        m = d.shape[0] * 2 - 2
        d = torch.tensor(self.sort_d(d))
        d_with_internals = torch.zeros((m, m))
        d_with_internals[: d.shape[0], : d.shape[0]] = d
        return d_with_internals

    def episode(self, d_list, n_taxa):
        batch_size = len(d_list)
        self.n_taxa = n_taxa
        self.m = self.n_taxa * 2 - 2
        pool = mp.Pool(self.num_procs)
        results = pool.map(self.make_d_batch, d_list)
        pool.close()
        pool.join()
        d = torch.stack(results).to(torch.float).to(self.device)

        adj_mat, size_mask, initial_mask, d_mask = self.initial_mats(d, batch_size)
        trajectory = []
        tau, idxs = None, None

        for i in range(3, self.n_taxa):
            ad_mask, mask = self.get_masks(adj_mat)
            tau, tau_mask = self.get_taus(adj_mat,  tau, idxs)
            state = adj_mat, ad_mask, d, d_mask, size_mask, initial_mask, mask, tau, tau_mask, None
            probs, l_probs = self.net(state)

            # y, _ = self.net(adj_mat.unsqueeze(0), self.d.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0))
            prob_dist = torch.distributions.Categorical(probs)  # probs should be of size batch x classes
            action = prob_dist.sample()
            trajectory.append(l_probs)
            idxs = torch.stack([torch.div(action, self.m, rounding_mode='trunc'), action % self.m]).T.to(self.device)
            adj_mat = self.add_node(adj_mat, idxs, new_node_idx=i, n=self.n_taxa)
            self.adj_mats.append(adj_mat.to("cpu").numpy())

        self.solution = self.adj_mats[-1].astype(int)
        self.obj_val = self.compute_obj_val_from_adj_mat(self.solution, self.d.to('cpu').numpy(), self.n_taxa)
        l_probs = torch.vstack(trajectory).flatten()
        l_probs = l_probs[l_probs > - 9e15]
        self.loss = sum(-self.obj_val * l_probs)
        self.optimiser.zero_grad()
        self.loss.backward()
        self.optimiser.step()

    def initial_mats(self, d, batch_size):
        adj_mat = self.initial_step_mats(batch_size)
        size_mask = torch.ones_like(d)
        initial_mask = torch.zeros((batch_size, self.m, 2)).to(self.device)  # ********************************************
        initial_mask[:, :, 0] = torch.tensor([1 if i < self.n_taxa else 0 for i in range(self.m)]).to(self.device)
        initial_mask[:, :, 1] = torch.tensor([1 if i >= self.n_taxa else 0 for i in range(self.m)]).to(self.device)
        d_mask = copy.deepcopy(d)
        d_mask[d_mask > 0] = 1

        return adj_mat, size_mask, initial_mask, d_mask

    def initial_step_mats(self, batch_size):
        adj_mat = torch.zeros((batch_size, self.m, self.m)).to(self.device)
        adj_mat[:, 0, self.n_taxa] = adj_mat[:, self.n_taxa, 0] = 1
        adj_mat[:, 1, self.n_taxa] = adj_mat[:, self.n_taxa, 1] = 1
        adj_mat[:, 2, self.n_taxa] = adj_mat[:, self.n_taxa, 2] = 1
        return adj_mat

    def get_masks(self, adj_mat):
        batch_size, m, _ = adj_mat.shape
        ad_mask = torch.zeros((batch_size, m, 2)).to(self.device)
        a = torch.sum(adj_mat, dim=-1)
        ad_mask[:, :, 0] = a
        ad_mask[ad_mask > 0] = 1
        ad_mask[:, :, 1] = torch.abs(ad_mask[:, :, 0] - 1)
        mask = torch.stack([torch.triu(adj_mat[i, :, :]) for i in range(batch_size)])
        return ad_mask, mask

    def get_taus(self, adj_mat, tau, idxs):
        n = self.n_taxa
        if tau is None:
            tau = torch.zeros_like(adj_mat)
            tau[:, 0, 1] = tau[:, 1, 0] = tau[:, 0, 2] = tau[:, 2, 0] = tau[:, 1, 2] = tau[:, 2, 1] = 2
            tau[:, 0, n] = tau[:, n, 0] = tau[:, n, 2] = tau[:, 2, n] = tau[:, 1, n] = tau[:, n, 1] = 1
            tau_mask = copy.deepcopy(tau)
            tau_mask[tau_mask > 0] = 1
            return tau, tau_mask
        else:
            pass

        @staticmethod
        def add_nodes(adj_mat, idxs, new_node_idx, n):
            adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
            adj_mat[idxs[0], n + new_node_idx - 2] = adj_mat[
                n + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
            adj_mat[idxs[1], n + new_node_idx - 2] = adj_mat[
                n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
            adj_mat[new_node_idx, n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, new_node_idx] = 1  # attach new

            return adj_mat




