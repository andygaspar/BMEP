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

batch_size = train_params["batch_size"]


dgn = net_manager.make_network()

path = 'Data_/csv_'
filenames = sorted(next(os.walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

m = mats[3]
dim_dataset = m.shape[0]

optimizer = optim.Adam(dgn.parameters(), lr=10 ** train_params["lr"], weight_decay=10 ** train_params["weight_decay"])

episodes = 1_000
batch_size = 64

pol = PolicyGradientBatchEpisode(dgn, optimizer)

directory, best_mean_difference = None, 10

for episode in range(1, episodes + 1):
    n_taxa = np.random.choice(range(10, 12))
    d_list = []
    for _ in range(batch_size):

        idx = random.sample(range(dim_dataset), k=n_taxa)
        d_list.append(m[idx, :][:, idx])

    loss, difference_mean, better, equal = pol.episode(d_list, n_taxa)
    print(episode, episode * batch_size, "taxa ", n_taxa, "   loss ", loss, "   difference mean", difference_mean,
          "   better", better,  "   equal", equal)

    if episode > 100 and difference_mean < best_mean_difference:
        if directory is not None and save:
            shutil.rmtree(directory)
        directory = dgn.save_net(folder, difference_mean, params, supervised=False)
        best_mean_difference = difference_mean


