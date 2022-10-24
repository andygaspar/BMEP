import torch
import torch.nn as nn
import torch.autograd as autograd

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
        out = nn.functional.softmax(torch.mul(a, mat) - 9e15 * (1 - mat_mask), dim=-1)
        return out


class FE(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        super(FE, self).__init__()
        self.device = device
        self.fe = nn.Linear(h_dimension * 2 + hidden_dim, hidden_dim).to(self.device)
        self.fd = nn.Linear(1, hidden_dim).to(self.device)

    def forward(self, hi, hj, d):
        dd = d.view(d.shape[0], d.shape[1] ** 2, 1)
        d_ij = torch.tanh(self.fd(dd))
        out = torch.tanh(self.fe(torch.cat([hi, hj, d_ij], dim=-1)))
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, h_dimension, hidden_dim, num_heads, device):
        self.device = device
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.device = device
        self.alpha = nn.ModuleList([Message(h_dimension, hidden_dim , self.device)
                                    for _ in range(self.num_heads)])
        self.e = nn.ModuleList([FE(h_dimension, hidden_dim, self.device)
                                for _ in range(self.num_heads)])
        self.reduction = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)

    def forward(self, h, mat, mat_mask):
        heads = []
        for i in range(self.num_heads):
            hi, hj = self.i_j(h)
            alpha = self.alpha[i](hi, hj, mat, mat_mask).unsqueeze(-1)
            e = self.e[i](hi, hj, mat).view(mat.shape[0], mat.shape[1], mat.shape[2], -1)
            heads.append((alpha * e).sum(dim=-2))

        out = self.reduction(torch.cat(heads, dim=-1))
        return out

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
        return h


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


class GNN_TAU_MH(Network):
    def __init__(self, net_params, network=None):
        super().__init__(net_params["normalisation factor"])
        num_inputs, h_dimension, hidden_dim = net_params["num_inputs"], \
                                              net_params["h_dimension"], net_params["hidden_dim"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((10, 10)).to(self.device)

        self.rounds = net_params["num_messages"]
        self.num_heads = net_params['num heads']

        self.encoder = NodeEncoder(num_inputs, h_dimension, self.device)

        self.fd = nn.ModuleList([MultiHeadAttention(h_dimension, hidden_dim, self.num_heads, self.device)
                                 for _ in range(self.rounds)])
        self.ft = nn.ModuleList([MultiHeadAttention(h_dimension, hidden_dim, self.num_heads, self.device)
                                 for _ in range(self.rounds)])

        self.drop_out = net_params['drop out']

        self.fm1 = MessageNode(h_dimension, hidden_dim, self.drop_out, self.device)
        self.fm2 = MessageNode(h_dimension, hidden_dim, self.drop_out, self.device)
        self.fa = FA(h_dimension, hidden_dim, self.drop_out, self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, data):
        adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_masks, y = data
        h = self.encoder(initial_masks)
        taus[taus > 0] = 1 / taus[taus > 0]
        h = self.context_message(h, d_mats, d_masks, initial_masks)
        h = self.tau_message(h, taus, tau_masks, ad_masks)
        h = self.fa(h)

        y_h = torch.matmul(h, h.permute(0, 2, 1)) * masks - 9e15 * (1 - masks)
        mat_size = y_h.shape
        y_hat = nn.functional.softmax(y_h.view(mat_size[0], -1), dim=-1)

        return y_hat, h

    def context_message(self, h, d, d_mask, initial_mask):
        for i in range(self.rounds):
            m_1 = self.fd[i](h, d, d_mask)
            hd = initial_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_d = initial_mask[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm1(h, m_1) * hd + h * h_not_d
        return h

    def tau_message(self, h, taus, tau_masks, ad_masks):
        for i in range(self.rounds):
            m_2 = self.fd[i](h, taus, tau_masks)
            h_adj = ad_masks[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_adj = ad_masks[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm2(h, m_2) * h_adj + h * h_not_adj
        return h

    def compute_message(self, ):
        pass

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
