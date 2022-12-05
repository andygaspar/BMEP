import torch
import torch.nn as nn
import torch.nn.functional as F
from ML.Net.network import Network


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


softmax = nn.functional.softmax


class Embedding(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(Embedding, self).__init__()
        self.device = device
        self.embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)
        # self.embedding.apply(init_weights)

    def forward(self, x):
        h = self.embedding(x)
        return h


class LeafLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(LeafLayer, self).__init__()
        self.device = device
        self.fcv = nn.Linear(m, hidden_dim).to(self.device)
        self.fck = nn.Linear(m, hidden_dim).to(self.device)
        self.fck.apply(init_weights)
        self.fcq = nn.Linear(m, hidden_dim).to(self.device)
        self.fcout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)
        self.eta_g = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)

    def forward(self, d, h):

        # v = F.relu(self.fcv(h))
        # # print(v, "\ngggh")
        # q = torch.relu(self.fcq(h))
        # k = torch.relu(self.fck(h)).permute(0, 2, 1)
        v = self.fcv(h)
        # print(v, "\ngggh")
        q = self.fcq(h)
        k = self.fck(h).permute(0, 2, 1)
        att = torch.bmm(q, k) # * d
        # print(att)
        att = F.softmax(att, dim=2)
        # print(att)
        out = torch.bmm(att, v)

        # print(out)
        # h = F.leaky_relu(self.fcout(out))
        h = self.fcout(out)

        return h


class TreeLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(TreeLayer, self).__init__()
        self.device = device
        self.fcv = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)
        self.fck = nn.Linear(m, hidden_dim).to(self.device)
        self.fcq = nn.Linear(m, hidden_dim).to(self.device)
        self.fcout = nn.Linear(hidden_dim, m).to(self.device)
        self.eta_g = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)
        # self.eta_g.apply(init_weights)

    def forward(self, A, h):
        v = F.relu(self.fcv(h))
        q = F.relu(self.fcq(h))
        k = F.relu(self.fck(h)).permute(0, 2, 1)
        att = torch.bmm(q, k) * (-A)
        att = F.softmax(att, dim=2)

        out = torch.bmm(att, v)
        out = torch.add(out, v)
        h = F.relu(self.fcout(out))
        return h


class DGN_test_2(Network):
    def __init__(self, m, hidden_dim_f, hidden_dim_g, n_messages, composed=True, network=None):
        super().__init__()
        self.m = m
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_massages = n_messages
        self.leaf_layers = nn.ModuleList([LeafLayer(m, hidden_dim_f, self.device) for _ in range(n_messages)])
        self.tree_layers = nn.ModuleList([TreeLayer(m, hidden_dim_g, self.device) for _ in range(n_messages)])
        # self.gru = torch.nn.GRUCell(hidden_dim, hidden_dim).to(self.device)
        self.init_embedding = Embedding(m, 64, self.device)
        self.f = nn.Sequential(
            nn.Linear(m * 2, hidden_dim_f),
            nn.ReLU(),
            nn.Linear(hidden_dim_f, m)
        ).to(self.device)

        if network is not None:
            self.load_weights(network)
        # self.test_net = TestNet(num_inputs, self.device)

    def forward(self, A, d, x, mask):
        scaled_d = d
        h0 = torch.zeros((d.shape[0], d.shape[1], 2)).to(self.device)
        # h0 = torch.normal(0, 100, size=h0.shape).to(self.device)
        h0[:, :, 0] = torch.mean(scaled_d, dim=-1) * 10
        h0[:, :, 1] = torch.var(scaled_d, dim=-1) * 10
        # t = h0[:, :, 0] - torch.mean(h0, dim=1)
        #h0 -= torch.mean(h0, dim=1, keepdim=True)
        # h0 /= torch.std(h0, dim=1, keepdim=True)
        h = self.init_embedding(h0)
        # h = torch.matmul(d*10, h)
        # scaled_d = (d * 100)**2

        for i in range(self.n_massages):
            h = self.leaf_layers[i](scaled_d, h)
            # h = self.f(torch.cat([h, h_], dim=-1))
            # h_2 = self.tree_layers[i](A, h)
            # h = self.f(torch.cat([h, h_1, h_2], dim=-1))

        # y_hat = torch.matmul(h, h.permute(0, 2, 1)) * mask
        # mat_size = y_hat.shape
        #
        # y_hat = (y_hat.view(y_hat.shape[0], -1) / torch.sum(y_hat, dim=(-2, -1)).unsqueeze(1)).view(mat_size)
        # k = y_hat[y_hat > 0]
        # y_hat = torch.matmul(h, h.permute(0, 2, 1))
        # y_hat = torch.sigmoid(y_hat)
        y_hat = softmax(torch.sum(h, dim=-1), dim=-1).unsqueeze(1)
        return y_hat, h


