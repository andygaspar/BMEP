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
from test_mor_dims import sort_d

print(os.getcwd())

from Data_.Datasets.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from importlib.metadata import version

a100 = True if version('torch') == '1.9.0+cu111' else False
edge = False

folder = 'GNN_TAU_MH'
data_folder = '03-M18_5_9'
save = True

net_manager = NetworkManager(folder)
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

episodes = 10

for episode in range(episodes):
    dim = random.sample(range(6, 9), k=1)
    idx = random.sample(range(dim_dataset), k=dim)
    mat = sort_d(copy.deepcopy(m[idx, :][:, idx]))
    pol = PolicyGradientEpisode(mat, dgn, optimizer)

