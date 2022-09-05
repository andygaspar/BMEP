import json

import torch
from torch import nn

from Net.network import Network
from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.Nets.GNN2.gnn_2 import GNN_2
from Net.Nets.GNN_edge.gnn_edge import GNN_edge
from Net.Nets.GNNGRU.gnn_gru import GNN_GRU

def mse(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    return mse(out, yy)


def custom_loss_1(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    loss = out - yy
    loss[loss < 0] = - loss[loss < 0] ** 2 - 2 * loss[loss < 0]
    loss = torch.mean(loss)
    return loss


def custom_loss_2(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    step_num = (1 / torch.sum(adj_mats, dim=(1, 2)) / 2).view(-1, 1, 1).expand(-1, *adj_mats.shape[1:])[
        masks > 0].flatten()
    loss = out - yy
    loss[loss < 0] = - loss[loss < 0] ** 2 - 2 * loss[loss < 0]
    loss = torch.mean(loss + step_num)
    return loss


def custom_loss_3(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
    out, yy = output.view(output.shape[0], 10, 10)[masks > 0].flatten(), \
              y.view(y.shape[0], 10, 10)[masks > 0].flatten()
    step_num = (1 / torch.sum(adj_mats, dim=(1, 2)) / 2).view(-1, 1, 1).expand(-1, *adj_mats.shape[1:])[
        masks > 0].flatten()
    loss = torch.mean(nn.MSELoss(out, yy) + step_num)
    return loss


nets_dict = {
    'GNN': GNN,
    'GNN1': GNN_1,
    'GNN2': GNN_2,
    'GNN_edge': GNN_edge,
    'GNNGRU': GNN_GRU
}

criterion_dict = {
    'mse': mse,
    'cross': nn.CrossEntropyLoss,
    'custom_1': custom_loss_1,
    'custom_2': custom_loss_2
}


class NetworkManager:

    @staticmethod
    def make_network(folder):
        path = 'Net/Nets/' + folder + '/'
        with open(path + 'params.json', 'r') as json_file:
            params = json.load(json_file)
            print(params)

        train_params, net_params = params["train"], params["net"]

        dgn = nets_dict[folder](net_params=net_params)

        return dgn, train_params, net_params, folder

    @staticmethod
    def get_network(folder, file):
        path = 'Net/Nets/' + folder + '/' + file + '/'

        with open(path + 'params.json', 'r') as json_file:
            params = json.load(json_file)
            print(params)
        net_params = params
        comment = params['comment'] if 'comment' in params.keys() else ''

        dgn = nets_dict[folder](net_params=net_params, network=path + "weights.pt")
        return dgn, comment

    @staticmethod
    def compute_loss(criterion, output, data):

        # print(sum(output[output > 0.9]))
        # loss = criterion(output, y.float())
        return criterion_dict[criterion](output, data)