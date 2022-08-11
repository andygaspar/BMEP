import os
import json
import copy
import shutil
import time
import json

import numpy as np
import torch
import os

from Net.Nets.GNN_edge.gnn_edge import GNN_edge


from Data_.Dataset.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader

from Net.Nets.GNN1.gnn_1 import GNN_1
from importlib.metadata import version



class NetTrainer:
    def __init__(self, configs, params_file):
        self._configs = configs
        self._a100 = True if version('torch') == '1.9.0+cu111' else False

        with open(os.path.join(self._configs['path'], params_file), 'r') as json_file:
            params = json.load(json_file)

        self._train_params = params['train']
        self._net_params = params['net']

        self._n_calls = 0



    def __call__(self, x):
        n = x.shape[0]
        vals = []
        for i in range(n):
            print(x[i])

            ### update dicts
            vals_dict = ....
            self._train_params.update(vals_dict)
            ###

            vals.append(self.train_net())

        return np.array(vals).reshape((n, 1))


    def train_net(self):
        dgn = GNN_1(self._net_params)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_ = BMEP_Dataset(scale_d=self._net_params["scale_d"], start=self._train_params["start"], end=self._train_params["end"], a100=self._a100)
        batch_size = self._train_params["batch_size"]
        dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(dgn.parameters(), lr=self._train_params["lr"], weight_decay=self._train_params["weight_decay"])
        for epoch in range(train_params["epochs"]):
            loss = None
            for data in dataloader:
                adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
                optimizer.zero_grad()
                output, h = dgn(adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            errors, n_samples = 0.0, 0.0
            for data in dataloader:
                adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
                output, h = dgn(adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks)

                idx = torch.max(output, dim=-1)[1]
                if a100:
                    err = torch.nonzero(idx - y).shape[0]
                else:
                    prediction = torch.zeros_like(output)
                    idi = torch.tensor(range(idx.shape[0])).to(device)
                    prediction[idi, idx] = 1
                    err = torch.sum(torch.abs(prediction - y.view(y.shape[0], -1))) / 2

                errors += err
                n_samples += len(y)

        fitness = errors / n_samples
        dgn.save_net(self._configs['path'], fitness, f"evaluation_{self._n_calls}", self._net_params)

        self._n_calls += 1
        return fitness
