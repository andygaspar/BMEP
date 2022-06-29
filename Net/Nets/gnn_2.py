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
        embedding = F.leaky_relu(self.fc(x))
        return embedding


class Message(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(Message, self).__init__()
        self.f_alpha = nn.Linear(h_dimension * 2, hidden_dim).to(self.device)
        self.v = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, hi, hj, mat, mat_mask):
        a = self.f_alpha(torch.cat([hi, hj], dim=-1))
        a = self.v(a).view(mat.shape)
        alpha = F.softmax(torch.mul(a, mat) - 9e15 * (1 - mat_mask), dim=-1)
        return alpha


class FD(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(FD, self).__init__()
        self.fe = nn.Linear(h_dimension * 2 + hidden_dim, hidden_dim).to(self.device)
        self.fd = nn.Linear(1, hidden_dim).to(self.device)

    def forward(self, hi, hj, d):
        dd = d.view(d.shape[0], d.shape[1]**2, 1)
        d_ij = self.fd(dd)
        out = F.leaky_relu(self.fe(torch.cat([hi, hj, d_ij], dim=-1)))
        return out


class F_ALL(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(F_ALL, self).__init__()
        self.f = nn.Linear(h_dimension * 2, hidden_dim).to(self.device)

    def forward(self, hi, hj):
        out = F.leaky_relu(self.f(torch.cat([hi, hj], dim=-1)))
        return out


class MessageNode(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(MessageNode, self).__init__()
        self.fmn = nn.Linear(h_dimension + hidden_dim, h_dimension).to(self.device)

    def forward(self, h, m1):
        return F.leaky_relu(self.fmn(torch.cat([h, m1], dim=-1)))


class FA(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(FA, self).__init__()
        self.fc1 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, h_dimension).to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        q = self.fc2(x)
        return q


class GNN_2(Network):
    def __init__(self, num_inputs, h_dimension, hidden_dim, num_messages=3, network=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((10, 10)).to(self.device)

        self.rounds = num_messages

        self.encoder = NodeEncoder(num_inputs, h_dimension, self.device)

        self.fd = nn.ModuleList([FD(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.ft = nn.ModuleList([F_ALL(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.fall = nn.ModuleList([F_ALL(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])

        self.alpha_d = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_t = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_all = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])

        # self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True).to(self.device)

        self.fm1 = MessageNode(h_dimension, hidden_dim, self.device)
        self.fm2 = MessageNode(h_dimension, hidden_dim, self.device)
        self.fm3 = MessageNode(h_dimension, hidden_dim, self.device)
        self.fa = FA(h_dimension, hidden_dim, self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, adj_mats, d_mats, initial_masks, masks):
        d_mats = d_mats * 10
        d_mask = d_mats.clone()
        d_mask[d_mask > 0] = 1
        ones = torch.ones_like(adj_mats) # to change when different input size
        h = self.encoder(initial_masks)

        h = self.message_round(h, d_mats, d_mask, adj_mats, ones, self.rounds)

        h = self.fa(h)

        y_h = torch.matmul(h, h.permute(0, 2, 1)) * masks - 9e15 * (1 - masks)
        mat_size = y_h.shape
        y_hat = F.softmax(y_h.view(mat_size[0], -1), dim=-1)

        return y_hat, h

    def message_round(self, h, d, d_mask, adj_mats, ones, rounds):

        for i in range(rounds):
            hi, hj = self.i_j(h)
            alpha_d = self.alpha_d[i](hi, hj, d, d_mask).unsqueeze(-1)
            e_d = self.fd[i](hi, hj, d).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_1 = (alpha_d * e_d).sum(dim=-2)

            h = self.fm1(h, m_1)
            hi, hj = self.i_j(h)
            alpha_t = self.alpha_t[i](hi, hj, adj_mats, adj_mats).unsqueeze(-1)
            e_t = self.ft[i](hi, hj).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_2 = (alpha_t * e_t).sum(dim=-2)

            h = self.fm2(h, m_2)
            hi, hj = self.i_j(h)
            alpha_all = self.alpha_all[i](hi, hj, ones, ones).unsqueeze(-1)
            e_all = self.ft[i](hi, hj).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_3 = (alpha_all * e_all).sum(dim=-2)

            # h, _ = self.gru(h.view(in_shape), m.view(m_shape))
            # h = h.view(out_shape)
            h = self.fm3(h, m_3)

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