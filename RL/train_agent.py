import copy
import random
import shutil
import time
import json

import numpy as np
import torch
import os
from Net.network_manager import NetworkManager
from RL.policy_gradient import PolicyGradientEpisode


from Data_.Datasets.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from importlib.metadata import version

from RL.policy_gradient_batch import PolicyGradientBatchEpisode
from Solvers.SWA.swa_solver import SwaSolver

print(os.getcwd())


def sort_d(d):
    dist_sum = np.sum(d, axis=0)
    order = np.argsort(dist_sum)
    sorted_d = np.zeros_like(d)
    for i in order:
        for j in order:
            sorted_d[i, j] = d[order[i], order[j]]
    return sorted_d


a100 = True if version('torch') == '1.9.0+cu111' else False
edge = False

folder = 'GNN_TAU'
data_folder = '03-M18_5_9'
save = True

net_manager = NetworkManager(folder, supervised=False)
params = net_manager.get_params()
train_params, net_params = params["train"], params["net"]
train_params["train data"] = data_folder

criterion = train_params["criterion"]
cross_entropy = True if criterion == "cross" else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_ = BMEP_Dataset(folder_name=data_folder, scale_d=net_params["scale_d"], start=train_params["start"],
                     end=train_params["end"], a100=a100)
batch_size = train_params["batch_size"]
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)

dgn = net_manager.make_network(normalisation_factor=data_.max_d_mat)

path = 'Data_/csv_'
filenames = sorted(next(os.walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

m = mats[3]
dim_dataset = m.shape[0]

optimizer = optim.Adam(dgn.parameters(), lr=10 ** train_params["lr"], weight_decay=10 ** train_params["weight_decay"])

episodes = 10_000
batch_size = 2

pol = PolicyGradientBatchEpisode(dgn, optimizer)

for episode in range(episodes):
    n_taxa = np.random.choice(range(6, 9))
    d_list = []
    for _ in range(batch_size):

        idx = random.sample(range(dim_dataset), k=n_taxa)
        d_list.append(sort_d(copy.deepcopy(m[idx, :][:, idx])))


    # swa = SwaSolver(d_list, n_taxa)
    # swa.solve()

    pol.episode(d_list, n_taxa)

    if episode % 10 == 0:
        print(episode, 'n_taxa', n_taxa, '  loss ', pol.loss.item(), '  agent obj', pol.obj_val,  '  swa', swa.obj_val,
              '  difference ', pol.obj_val - swa.obj_val)


# for episode in range(episodes):
#     dim = np.random.choice(range(6, 9))
#     idx = random.sample(range(dim_dataset), k=dim)
#     mat = sort_d(copy.deepcopy(m[idx, :][:, idx]))
#     d = np.zeros((dim*2-2, dim*2-2))
#     d[:dim, :dim] = mat
#
#     swa = SwaSolver(d)
#     swa.solve()
#     pol = PolicyGradientEpisode(d, dgn, optimizer)
#     pol.episode(swa.obj_val)
#
#     if episode % 10 == 0:
#         print(episode, 'dim', dim, '  loss ', pol.loss.item(), '  agent obj', pol.obj_val,  '  swa', swa.obj_val,
#               '  difference ', pol.obj_val - swa.obj_val)
