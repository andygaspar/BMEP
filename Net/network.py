import abc
import copy
import json
import os

import networkx as nx
import numpy as np
from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, net_params):
        super(Network, self).__init__()
        self.normalisation_factor = net_params["normalisation factor"]
        self.loss = 0

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))

    def save_net(self, folder: str, score: float, params: dict, prefix="", supervised=True):
        if supervised:
            new_folder = 'Net/Nets/Supervised/' + folder + '/' + prefix + "_" + str(int(score * 1000) / 1000) + "_0"
        else:
            new_folder = 'Net/Nets/RlNets/' + folder + '/' + prefix + "_" + str(int(score * 1000) / 1000) + "_0"
        while os.path.isdir(new_folder):
            new_folder += "i"
        os.mkdir(new_folder)
        score = "best_loss" if supervised else "best mean difference"
        params["net"][score] = score
        params["net"]["normalisation factor"] = self.normalisation_factor
        with open(new_folder + '/params.json', 'w') as outfile:
            json.dump(params, outfile)
        torch.save(self.state_dict(), new_folder + '/weights.pt')
        return new_folder

    def get_net_input(self, adj_mat, d, tau, m, n_taxa, step, n_problems=1):
        pass


class Gat(Network):

    def __init__(self, net_params):
        super().__init__(net_params)
        self.d = None
        self.initial_mask = None
        self.d_mask = None
        self.size_mask = None

    @staticmethod
    def get_tau_mask(tau):
        tau_mask = copy.deepcopy(tau)
        tau_mask[tau_mask > 0] = 1
        return tau_mask

    def get_masks(self, adj_mat):
        ad_mask = torch.zeros((adj_mat.shape[0], 2)).to(self.device)
        a = torch.sum(adj_mat, dim=-1)
        ad_mask[:, 0] = a
        ad_mask[ad_mask > 0] = 1
        ad_mask[:, 1] = torch.abs(ad_mask[:, 0] - 1)
        mask = torch.triu(adj_mat)
        return ad_mask, mask

    def get_net_input(self, adj_mat, d, tau, m, n_taxa, step, n_problems=1):
        if step == 3:
            self.d = d
            self.size_mask = torch.ones_like(self.d)
            self.d_mask = copy.deepcopy(self.d)
            self.d_mask[self.d_mask > 0] = 1
            self.initial_mask = torch.zeros((m, 2)).to(self.device)
            self.initial_mask[:, 0] = torch.tensor([1 if i < n_taxa else 0 for i in range(m)]).to(self.device)
            self.initial_mask[:, 1] = torch.tensor([1 if i >= n_taxa else 0 for i in range(m)]).to(self.device)
        tau_mask = self.get_tau_mask(tau)
        ad_mask, mask = self.get_masks(adj_mat)
        return (adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), d.unsqueeze(0), self.d_mask.unsqueeze(0),
                self.size_mask.unsqueeze(0), self.initial_mask.unsqueeze(0), mask.unsqueeze(0),
                tau.unsqueeze(0),
                tau_mask.unsqueeze(0))


class EGAT(Network):

    def __init__(self, net_params):
        super().__init__(net_params)
        self.d = None
        self.initial_mask = None
        self.d_mask = None
        self.size_mask = None
        self.taxa_inputs, self.internal_inputs, self.edge_inputs = \
            net_params["taxa_inputs"], net_params["internal_inputs"], net_params["edge_inputs"]
        self.taxa_embeddings = None
        self.internal_embeddings = None
        self.message_embeddings = None
        self.current_mask = None
        self.active_nodes_mask = None

    @staticmethod
    def get_min_max(tens, expanded_size):
        if tens.shape[-1] == 1:
            return tens
        else:
            max_vals = tens.max(dim=-1).values.view(-1, 1).expand(-1, expanded_size)
            min_vals = tens.min(dim=-1).values.view(-1, 1).expand(-1, expanded_size)
            return (tens - min_vals) / (max_vals - min_vals)

    def get_net_input(self, adj_mat, d, tau, m, n_taxa, step, n_problems=1):

        if len(adj_mat.shape) < 3:
            adj_mat = adj_mat.unsqueeze(0)
            d = d.unsqueeze(0)
            tau = tau.unsqueeze(0)
        if step == 3:
            self.current_mask = torch.zeros((n_problems, m, m)).to(self.device)
            self.active_nodes_mask = torch.zeros((n_problems, m, 2)).to(self.device)
            self.active_nodes_mask[:, :n_taxa, 0] = 1

            self.taxa_embeddings = torch.zeros((n_problems, m, self.taxa_inputs)).to(self.device)
            self.taxa_embeddings[:, :n_taxa, 0] = (
                    d[:, :n_taxa, :n_taxa] / self.normalisation_factor).mean(
                dim=-1)
            self.taxa_embeddings[:, :n_taxa, 1] = self.get_min_max(self.taxa_embeddings[:, :n_taxa, 0],
                                                                   n_taxa)
            self.taxa_embeddings[:, :n_taxa, 2] = (
                    d[:, :n_taxa, :n_taxa] / self.normalisation_factor).std(
                dim=-1)
            self.taxa_embeddings[:, :n_taxa, 3] = self.get_min_max(self.taxa_embeddings[:, :n_taxa, 2],
                                                                   n_taxa)

            self.internal_embeddings = torch.zeros((n_problems, m, self.internal_inputs)).to(self.device)

            message_d = copy.deepcopy(d) / self.normalisation_factor
            message_d[:, n_taxa:, :] = message_d[:, :, n_taxa:] = 1
            message_d = 1 - message_d
            min_vals = message_d[:, :n_taxa, :n_taxa].reshape(-1, n_taxa ** 2).min(dim=-1).values \
                .unsqueeze(-1).unsqueeze(-1).expand(-1, m, m)

            self.message_embeddings = torch.zeros((n_problems, m ** 2, self.edge_inputs)).to(self.device)
            self.message_embeddings[:, :, 0] = message_d.reshape(-1, m ** 2)
            self.message_embeddings[:, :, 1] = ((message_d - min_vals) / (1 - min_vals)) \
                .reshape(-1, m ** 2)
            self.message_embeddings[self.message_embeddings < 0] = 0

        self.current_mask[:, :n_taxa + step - 2, :n_taxa + step - 2] = 1
        self.active_nodes_mask[:, n_taxa: n_taxa + step - 2, 1] = 1

        self.taxa_embeddings[:, :, 4] = 1 / step
        self.taxa_embeddings[:, :, 5] = step / n_taxa
        self.taxa_embeddings[:, :step, 6] = (
                d[:, :step, :step] / (self.normalisation_factor * 2 ** tau[:, :step, :step])).sum(dim=-1)
        self.taxa_embeddings[:, :step, 7] = self.get_min_max(self.taxa_embeddings[:, :step, 6], step)

        self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 0] = 1 / step
        self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 1] = step / n_taxa
        mean_cur_taxa = self.taxa_embeddings[:, :step, 0].unsqueeze(1).repeat(1, step - 2, 1)
        std_cur_taxa = self.taxa_embeddings[:, :step, 1].unsqueeze(1).repeat(1, step - 2, 1)
        tau_cur_internal = tau[:, n_taxa: n_taxa + step - 2, :step]
        self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 2] = \
            (mean_cur_taxa / 2 ** tau_cur_internal).sum(dim=-1)
        self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 3] = \
            self.get_min_max(self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 2], step - 2)
        self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 4] = (
                std_cur_taxa / 2 ** tau_cur_internal).sum(dim=-1)
        self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 5] = \
            self.get_min_max(self.internal_embeddings[:, n_taxa: n_taxa + step - 2, 4], step - 2)

        self.internal_embeddings[:, n_taxa:, 6] = (tau[:, n_taxa:, :] / 2 ** tau[:, n_taxa:, :]).mean(dim=-1)
        self.internal_embeddings[:, n_taxa:, 7] = \
            self.get_min_max(self.internal_embeddings[:, n_taxa:, 6], m - n_taxa)

        message_i = tau + 1
        message_i[message_i > 1] = 4 / 10 + 1 / message_i[message_i > 1]
        message_i[:, step: n_taxa, :] = message_i[:, :, step: n_taxa] = \
            message_i[:, :, n_taxa + step - 2:] = message_i[:, n_taxa + step - 2:, :] = 0
        self.message_embeddings[:, :, 2] = message_i.reshape(-1, m ** 2)
        max_val = tau.reshape(-1, m ** 2).max(dim=-1).values.view(-1, 1, 1).expand(-1, m, m)
        message_linear = 1 - tau / max_val
        self.message_embeddings[:, :, 3] = message_linear.reshape(-1, m ** 2)

        action_mask = torch.stack([torch.triu(adj_mat[i, :, :]) for i in range(adj_mat.shape[0])])

        return self.taxa_embeddings, self.internal_embeddings, self.message_embeddings, \
               self.current_mask, self.active_nodes_mask, action_mask
