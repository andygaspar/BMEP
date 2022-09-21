import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from Net.network import Network


class NodeEncoder(nn.Module):
    def __init__(self, din, h_dimension, device=None):
        super(NodeEncoder, self).__init__()
        self.device = device
        self.fc = nn.Linear(din, h_dimension).to(self.device)

    def forward(self, x):
        embedding = torch.tanh(self.fc(x))
        return embedding


class Message(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(Message, self).__init__()
        self.f_alpha = nn.Linear(h_dimension * 2, hidden_dim).to(self.device)
        self.v = nn.Linear(hidden_dim, 1).to(self.device)
        # self.v = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, 1)).to(self.device)

    def forward(self, hi, hj, mat, mat_mask):
        a = torch.tanh(self.f_alpha(torch.cat([hi, hj], dim=-1)))  # messo lrelu
        a = torch.tanh(self.v(a).view(mat.shape))  # messo lrelu
        alpha = F.softmax(torch.mul(a, mat) - 9e15 * (1 - mat_mask), dim=-1)
        return alpha


class FD(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(FD, self).__init__()
        self.fe = nn.Linear(h_dimension * 2 + hidden_dim, hidden_dim).to(self.device)
        self.fd = nn.Linear(1, hidden_dim).to(self.device)

    def forward(self, hi, hj, d):
        dd = d.view(d.shape[0], d.shape[1] ** 2, 1)
        d_ij = torch.tanh(self.fd(dd))  # messo lrelu
        out = torch.tanh(self.fe(torch.cat([hi, hj, d_ij], dim=-1)))
        return out


class MessageNode(nn.Module):
    def __init__(self, h_dimension, hidden_dim, drop_out, device):
        self.device = device
        super(MessageNode, self).__init__()
        self.fmn1 = nn.Linear(h_dimension + hidden_dim, h_dimension).to(self.device)
        self.fmn2 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.drop_out = drop_out

    def forward(self, h, m1):
        h = torch.tanh(self.fmn1(torch.cat([h, m1], dim=-1)))
        h = nn.functional.dropout(h, p=self.drop_out)
        return torch.tanh(h)


class FA(nn.Module):
    def __init__(self, h_dimension, hidden_dim, drop_out, device):
        self.device = device
        super(FA, self).__init__()
        self.fc1 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, h_dimension).to(self.device)
        self.drop_out = drop_out

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # x = nn.functional.dropout(x, p=self.drop_out)
        q = self.fc2(x)
        return q


class GNN_TAU(Network):
    def __init__(self, net_params, network=None):
        super().__init__(net_params["normalisation factor"])
        num_inputs, h_dimension, hidden_dim, num_messages = net_params["num_inputs"], net_params["h_dimension"], \
                                                            net_params["hidden_dim"], net_params["num_messages"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((10, 10)).to(self.device)

        self.rounds = num_messages

        self.encoder = NodeEncoder(num_inputs, h_dimension, self.device)

        self.fd = nn.ModuleList([FD(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.ft = nn.ModuleList([FD(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])

        self.alpha_d = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_t = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])

        self.drop_out = net_params['drop out']

        self.fm1 = MessageNode(h_dimension, hidden_dim, self.drop_out, self.device)
        self.fm2 = MessageNode(h_dimension, hidden_dim, self.drop_out, self.device)
        self.fa = FA(h_dimension, hidden_dim, self.drop_out, self.device)



        if network is not None:
            self.load_weights(network)

    def forward(self, data):
        adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_masks, y = data
        d_mats_ = d_mats / self.normalisation_factor
        h = self.encoder(initial_masks)
        taus[taus > 0] = 1 / taus[taus > 0]
        h = self.context_message(h, d_mats_, d_masks, initial_masks, 3)
        h = self.tau_message(h, taus, tau_masks, ad_masks, 3)
        h = self.fa(h)

        y_h = torch.matmul(h, h.permute(0, 2, 1)) * masks - 9e15 * (1 - masks)
        mat_size = y_h.shape
        y_hat = F.softmax(y_h.view(mat_size[0], -1), dim=-1)

        return y_hat, h

    def context_message(self, h, d, d_mask, initial_mask, rounds):
        for i in range(rounds):
            hi, hj = self.i_j(h)
            alpha_d = self.alpha_d[i](hi, hj, d, d_mask).unsqueeze(-1)
            e_d = self.fd[i](hi, hj, d).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_1 = (alpha_d * e_d).sum(dim=-2)
            hd = initial_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_d = initial_mask[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm1(h, m_1) * hd + h * h_not_d

        return h

    def tau_message(self, h, taus, tau_masks, ad_masks, rounds):
        for i in range(rounds):
            hi, hj = self.i_j(h)
            alpha_t = self.alpha_t[i](hi, hj, taus, tau_masks).unsqueeze(-1)
            e_d = self.ft[i](hi, hj, taus).view(taus.shape[0], taus.shape[1], taus.shape[2], -1)
            m_2 = (alpha_t * e_d).sum(dim=-2)
            h_adj = ad_masks[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_adj = ad_masks[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm2(h, m_2) * h_adj + h * h_not_adj
        return h

    @staticmethod
    def i_j(h):
        idx = torch.tensor(range(h.shape[1]))
        idxs = torch.cartesian_prod(idx, idx)
        idxs = idxs[[i for i in range(idxs.shape[0])]]
        # hi = h[:, idxs[:, 0]].view((h.shape[0], e_shape[1], e_shape[2], -1))
        # hj = h[:, idxs[:, 1]].view((h.shape[0], e_shape[1], e_shape[2], -1))
        hi = h[:, idxs[:, 0]]
        hj = h[:, idxs[:, 1]]

        return hi, hj
