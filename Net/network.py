import abc
import copy
import json
import os

import networkx as nx
import numpy as np
from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, normalisation_factor):
        super(Network, self).__init__()
        self.normalisation_factor = normalisation_factor
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


class Gat(Network):

    def __init__(self, normalisation_factor):
        super().__init__(normalisation_factor)
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

    def get_net_input(self, adj_mat, d, tau, m, n_taxa, initial=False):
        if initial:
            self.d = d
            self.size_mask = torch.ones_like(self.d)
            self.d_mask = copy.deepcopy(self.d)
            self.d_mask[self.d_mask > 0] = 1
            self.initial_mask = torch.zeros((self.m, 2)).to(self.device)
            self.initial_mask[:, 0] = torch.tensor([1 if i < n_taxa else 0 for i in range(m)]).to(self.device)
            self.initial_mask[:, 1] = torch.tensor([1 if i >= n_taxa else 0 for i in range(m)]).to(self.device)
        tau_mask = self.get_tau_mask(tau)
        ad_mask, mask = self.get_masks()
        return (adj_mat.unsqueeze(0), ad_mask.unsqueeze(0), d.unsqueeze(0), self.d_mask.unsqueeze(0),
                self.size_mask.unsqueeze(0), self.initial_mask.unsqueeze(0), mask.unsqueeze(0),
                tau.unsqueeze(0),
                tau_mask.unsqueeze(0), None)
