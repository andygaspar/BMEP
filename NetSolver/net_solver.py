import copy

import torch.nn.init


class NetSolver:

    def __init__(self, d, net):
        self.d = d
        # adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m = d.shape[0]
        self.n = (self.m + 2) // 2

        self.adj_mats = []
        self.solution = None

    def solve(self):
        adj_mat = torch.zeros_like(self.d).to(self.device)
        size_mask = torch.ones_like(self.d)
        adj_mat[0, self.n] = adj_mat[self.n, 0] = 1
        adj_mat[1, self.n] = adj_mat[self.n, 1] = 1
        adj_mat[2, self.n] = adj_mat[self.n, 2] = 1
        initial_mask = torch.zeros((self.m, 2)).to(self.device)
        initial_mask[:, 0] = torch.tensor([1 if i < self.n else 0 for i in range(self.m)]).to(self.device)
        initial_mask[:, 1] = torch.tensor([1 if i >= self.n else 0 for i in range(self.m)]).to(self.device)
        mask = torch.triu(adj_mat)
        with torch.no_grad():
            for i in range(3, self.n):
                ad_mask = torch.zeros((self.d.shape[0], 2)).to(self.device)
                a = torch.sum(adj_mat, dim=-1)
                ad_mask[:, 0] = a
                ad_mask[ad_mask > 0] = 1
                ad_mask[:, 1] = torch.abs(ad_mask[:, 0] - 1)
                d_mask = copy.deepcopy(self.d)
                d_mask[d_mask > 0] = 1
                y, _ = self.net(adj_mat.unsqueeze(0), ad_mask.unsqueeze(0),  self.d.unsqueeze(0), d_mask.unsqueeze(0),
                                size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0))
                # y, _ = self.net(adj_mat.unsqueeze(0), self.d.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0))
                y = y.squeeze(0).view(self.d.shape)
                a_max = torch.argmax(y)
                idxs = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)
                adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
                adj_mat[idxs[0], self.n + i - 2] = adj_mat[self.n + i - 2, idxs[0]] = 1  # reattach selected to new
                adj_mat[idxs[1], self.n + i - 2] = adj_mat[self.n + i - 2, idxs[1]] = 1  # reattach selected to new
                adj_mat[i, self.n + i - 2] = adj_mat[self.n + i - 2, i] = 1  # attach new
                mask = torch.triu(adj_mat)
                self.adj_mats.append(adj_mat.to("cpu").numpy())

        self.solution = self.adj_mats[-1].astype(int)
