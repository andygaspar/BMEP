import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from Net.network import Network


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


softmax = nn.functional.softmax


class LeafLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(LeafLayer, self).__init__()
        self.device = device
        self.eta_f = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)
        # self.eta_f.apply(init_weights)

    def forward(self, A, d, h, x):
        mat_size = h.shape
        k = torch.bmm(d, h)
        max_d = torch.max(d.view(d.shape[0], 1, -1), dim=-1)[0]
        k = (k.view(h.shape[0], -1) / max_d).view(mat_size)
        k = self.eta_f(k)
        k = k * x[:, :, -2].view((x.shape[0], -1, 1))
        h = h * x[:, :, -1].view((x.shape[0], -1, 1)) + k
        return h


class NodesLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(NodesLayer, self).__init__()
        self.device = device
        self.eta_g = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)
        # self.eta_g.apply(init_weights)

    def forward(self, A, h):
        k = torch.bmm(A, h) / 3
        k = self.eta_g(k)
        A_mask = torch.max(A, dim=-1)[0].unsqueeze(-1)
        h = k * A_mask + h * (A_mask - 1) * (-1)
        return h


class InnerLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(InnerLayer, self).__init__()
        self.device = device
        self.fcv = nn.Linear(m, hidden_dim).to(self.device)
        self.fck = nn.Linear(m, hidden_dim).to(self.device)
        self.fcq = nn.Linear(m, hidden_dim).to(self.device)
        self.fcout = nn.Linear(hidden_dim, m).to(self.device)
        self.eta_g = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)
        # self.eta_g.apply(init_weights)

    def forward(self, h, x):
        v = F.relu(self.fcv(h))
        # q = F.relu(self.fcq(h)) * x[:, :, -1].view((x.shape[0], -1, 1))
        # k = (F.relu(self.fck(h)) * x[:, :, -2].view((x.shape[0], -1, 1))).permute(0, 2, 1)
        # att = F.softmax(torch.bmm(q, k), dim=2) * x[:, :, -1].view((x.shape[0], -1, 1))
        # out = torch.bmm(att, v)
        # out = F.relu(self.fcout(out)) * x[:, :, -1].view((x.shape[0], -1, 1))
        # h = h * x[:, :, -2].view((x.shape[0], -1, 1)) + out
        q = F.relu(self.fcq(h))
        k = F.relu(self.fck(h)).permute(0, 2, 1)
        att = F.softmax(torch.bmm(q, k), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        h = F.relu(self.fcout(out))
        return h


class DGN_test(Network):
    def __init__(self, m, hidden_dim_f, hidden_dim_g, n_messages, composed=True, network=None):
        super().__init__()
        self.m = m
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_massages = n_messages
        self.leaf_layers = [LeafLayer(m, hidden_dim_f, self.device) for _ in range(n_messages)]
        self.nodes_layers = [NodesLayer(m, hidden_dim_g, self.device) for _ in range(n_messages)]
        self.inner_layers = [InnerLayer(m, hidden_dim_f, self.device) for _ in range(n_messages)]
        self.t_l = torch.normal(0, 1, size=(1, m), dtype=torch.float).to(self.device)
        self.t_f = torch.normal(0, 1, size=(1, m), dtype=torch.float).to(self.device)
        # self.layers = nn.ModuleList([val for pair in zip(self.leaf_layers, self.nodes_layers) for val in pair]
        #                             if composed else self.leaf_layers + self.nodes_layers)
        self.leaf_layers = nn.ModuleList(self.leaf_layers + self.nodes_layers + self.inner_layers)
        self.attention = nn.ModuleList([AttModel(m, hidden_dim_f, self.device) for _ in range(n_messages)])
        # self.net = nn.Sequential(*layers)

        if network is not None:
            self.load_weights(network)
        # self.test_net = TestNet(num_inputs, self.device)

    def forward(self, A, d, x, mask):

        # h0 = torch.zeros((d.shape[0], d.shape[1], 2)).to(self.device)
        # h0[:, :, 0] = torch.mean(d, dim=-1)
        # h0[:, :, 1] = torch.var(d, dim=-1)
        # k = torch.matmul(mask.permute(0, 2, 1), h0)
        leaf_adj = torch.matmul(mask.permute(0, 2, 1), torch.diag_embed(x[:, :, -2]))
        leaf_adj = leaf_adj + leaf_adj.permute(0, 2, 1)
        inner_adj = torch.matmul(mask.permute(0, 2, 1), torch.diag_embed(x[:, :, -1]))
        inner_adj = inner_adj + inner_adj.permute(0, 2, 1)
        b = (torch.sum(A, dim=-1) + 1000) % 1000

        h = self.t_l * x[:, :, -2].view((x.shape[0], -1, 1)) + self.t_f * x[:, :, -1].view((x.shape[0], -1, 1))
        # h = self.net(A, d, h0)
        for i in range(self.n_massages):
            h = self.leaf_layers[i](A, d, h, x)
            # h = self.inner_layers[i](A, d, h, x)
            h = self.inner_layers[i](h, x)
            h = self.nodes_layers[i](A, h)

        # y_hat = torch.matmul(h, h.permute(0, 2, 1)) * mask
        # mat_size = y_hat.shape
        #
        # y_hat = (y_hat.view(y_hat.shape[0], -1) / torch.sum(y_hat, dim=(-2, -1)).unsqueeze(1)).view(mat_size)
        # k = y_hat[y_hat > 0]
        y_hat = torch.matmul(h, h.permute(0, 2, 1))
        y_hat = torch.sigmoid(y_hat)
        return y_hat, h


class AttModel(nn.Module):
    def __init__(self, din, hidden_dim, device):
        self.device = device
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim).to(self.device)
        self.fck = nn.Linear(din, hidden_dim).to(self.device)
        self.fcq = nn.Linear(din, hidden_dim).to(self.device)
        self.fcout = nn.Linear(hidden_dim, din).to(self.device)

    def forward(self, h):
        v = F.relu(self.fcv(h))
        q = F.relu(self.fcq(h))
        k = F.relu(self.fck(h)).permute(0, 2, 1)
        att = F.softmax(torch.bmm(q, k), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))
        return out