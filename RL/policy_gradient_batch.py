import copy
import multiprocessing as mp
import numpy as np
import torch.nn.init

from Solvers.solver import Solver


class PolicyGradientBatchEpisode(Solver):

    def __init__(self, d_list, net, optim):
        self.batch_size = len(d_list)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        d_with_internals = torch.zeros((self.batch_size, self.m, self.m)).to(self.device)
        num_procs = mp.cpu_count()
        pool = mp.Pool(num_procs)
        results = pool.map(self.sort_d, d_list)
        # d_with_internals[:self.n_taxa, :self.n_taxa] = d if type(d) == torch.Tensor else torch.tensor(d)
        # d_list_sorted = [self.]
        # d = torch.tensor(d).to(torch.float).to(self.device)
        d = None
        super().__init__(d)        # adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks
        self.net = net

        self.optimiser = optim
        self.adj_mats = []
        self.loss = None

    def episode(self):
        adj_mat, size_mask, initial_mask, d_mask = self.initial_mats()
        trajectory = []

        for i in range(3, self.n_taxa):
            ad_mask, mask = self.get_masks(adj_mat)
            tau, tau_mask = self.get_tau_tensor(adj_mat, self.device)
            state = adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), self.d.unsqueeze(0), d_mask.unsqueeze(0), \
                    size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0), tau.unsqueeze(0), \
                    tau_mask.unsqueeze(0), None
            probs, l_probs = self.net(state)

            # y, _ = self.net(adj_mat.unsqueeze(0), self.d.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0))
            a_max = torch.argmax(probs.squeeze(0))
            trajectory.append(l_probs)
            idxs = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)
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

    def initial_mats(self):
        adj_mat = self.initial_step_mats(self.device)
        size_mask = torch.ones_like(self.d)
        initial_mask = torch.zeros((self.m, 2)).to(self.device)
        initial_mask[:, 0] = torch.tensor([1 if i < self.n_taxa else 0 for i in range(self.m)]).to(self.device)
        initial_mask[:, 1] = torch.tensor([1 if i >= self.n_taxa else 0 for i in range(self.m)]).to(self.device)
        d_mask = copy.deepcopy(self.d)
        d_mask[d_mask > 0] = 1

        return adj_mat, size_mask, initial_mask, d_mask

    def initial_step_mats(self, device):
        adj_mat = np.zeros((self.batch_size, self.m, self.m)) if device is None else torch.zeros((self.m, self.m)).to(device)
        adj_mat[:, 0, self.n_taxa] = adj_mat[self.n_taxa, 0] = 1
        adj_mat[:, 1, self.n_taxa] = adj_mat[self.n_taxa, 1] = 1
        adj_mat[:, 2, self.n_taxa] = adj_mat[self.n_taxa, 2] = 1

    def get_masks(self, adj_mat):
        ad_mask = torch.zeros((self.d.shape[0], 2)).to(self.device)
        a = torch.sum(adj_mat, dim=-1)
        ad_mask[:, 0] = a
        ad_mask[ad_mask > 0] = 1
        ad_mask[:, 1] = torch.abs(ad_mask[:, 0] - 1)
        mask = torch.triu(adj_mat)
        return ad_mask, mask




