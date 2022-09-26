import json

import torch
from torch import nn

from Net.Nets.GNN_TAU.gnn_tau import GNN_TAU
from Net.Nets.GNN_TAU_MH.gnn_tau_multi_head import GNN_TAU_MH
from Net.network import Network
from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.Nets.GNN2.gnn_2 import GNN_2
from Net.Nets.GNN_edge.gnn_edge import GNN_edge
from Net.Nets.GNN_GRU.gnn_gru import GNN_GRU


def mse(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_maks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    return mse(out, yy)


def custom_loss_1(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_maks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    loss = out - yy
    loss[loss < 0] = - loss[loss < 0] ** 2 - 2 * loss[loss < 0]
    loss = torch.mean(loss)
    return loss


def custom_loss_2(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_maks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    step_num = (1 / torch.sum(adj_mats, dim=(1, 2)) / 2).view(-1, 1, 1).expand(-1, *adj_mats.shape[1:])[
        masks > 0].flatten()
    loss = out - yy
    loss[loss < 0] = - loss[loss < 0] ** 2 - 2 * loss[loss < 0]
    loss = torch.mean(loss + step_num)
    return loss


def custom_loss_3(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_maks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    step_num = (1 / torch.sum(adj_mats, dim=(1, 2)) / 2).view(-1, 1, 1).expand(-1, *adj_mats.shape[1:])[
        masks > 0].flatten()
    loss = torch.mean(nn.MSELoss(out, yy) + step_num)
    return loss


def cross_entropy(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_maks, y = data
    loss = nn.CrossEntropyLoss()
    return loss(output, y)


nets_dict = {
    'GNN': GNN,
    'GNN1': GNN_1,
    'GNN2': GNN_2,
    'GNN_edge': GNN_edge,
    'GNN_GRU': GNN_GRU,
    'GNN_TAU': GNN_TAU,
    'GNN_TAU_MH': GNN_TAU_MH
}

criterion_dict = {
    'mse': mse,
    'cross': cross_entropy,
    'custom_1': custom_loss_1,
    'custom_2': custom_loss_2
}


class NetworkManager:

    def __init__(self, folder, file=None):
        self.folder = folder
        self.file = file
        self.path = 'Net/Nets/' + folder + '/'
        if self.file is not None:
            self.path += self.file + '/'

        with open(self.path + 'params.json', 'r') as json_file:
            self.params = json.load(json_file)

        self.train_params, self.net_params = self.params["train"], self.params["net"]

    def make_network(self, normalisation_factor):
        self.print_info()
        self.net_params["normalisation factor"] = normalisation_factor
        dgn = nets_dict[self.folder](net_params=self.net_params)

        return dgn

    def get_network(self):
        if self.file is not None:
            self.print_info()
            dgn = nets_dict[self.folder](net_params=self.net_params, network=self.path + "weights.pt")
            return dgn
        else:
            return None

    def get_params(self):
        return self.params

    @staticmethod
    def compute_loss(criterion, output, data):
        # print(sum(output[output > 0.9]))
        # loss = criterion(output, y.float())
        return criterion_dict[criterion](output, data)

    def print_info(self):
        print("Training")
        for key in self.train_params:
            print(key + ':', self.train_params[key])
        print("Network")
        for key in self.net_params:
            print(key + ':', self.net_params[key])
        if 'comment' in list(self.params.keys()):
            print('comment:', self.params['comment'])
