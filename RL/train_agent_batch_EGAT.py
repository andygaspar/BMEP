import copy
import random
import shutil
import time
import json

import numpy as np
import torch
import os
from Net.network_manager import NetworkManager
from RL.Batches.batch_EGAT import BatchEGAT
from RL.policy_gradient import PolicyGradientEpisode


from Data_.Datasets.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from importlib.metadata import version

from RL.policy_gradient_batch import PolicyGradientBatchEpisode
from RL.policy_gradient_batch_EGAT import PolicyGradientEGAT
from RL.policy_gradient_batch_RM import PolicyGradientBatch
from RL.Batches.batch import Batch
from RL.trainer import Trainer
from Solvers.SWA.swa_solver import SwaSolver


a100 = True if version('torch') == '1.9.0+cu111' else False
edge = False

folder = 'EGAT_RL'
save = True
dataset_num = 2


path = 'Data_/csv_'
filenames = sorted(next(os.walk(path), (None, None, []))[2])
mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

m = mats[dataset_num]
dim_dataset = m.shape[0]
normalisation_factor = np.max(m)
print(normalisation_factor)

net_manager = NetworkManager(folder, supervised=False, normalisation_factor=normalisation_factor)
params = net_manager.get_params()
train_params, net_params = params["train"], params["net"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = train_params["batch_size"]


dgn = net_manager.make_network()
optimizer = optim.Adam(dgn.parameters(), lr=10 ** train_params["lr"], weight_decay=10 ** train_params["weight_decay"])


min_num_taxa, max_num_taxa = train_params["min_num_taxa"], train_params["max_num_taxa"]

runs = train_params["runs"]
episodes_in_run, episodes_in_parallel = train_params["episodes_in_run"], train_params["episodes_in_parallel"]

pol = PolicyGradientEGAT(dgn)

directory, best_mean_loss = None, 10**5

training_batch_size, training_loops = train_params['batch_size'], train_params['training_loops']
trainer = Trainer(dgn, optimizer, training_batch_size, training_loops)

num_instances = 0

for run in range(runs):
    n_taxa_list = np.random.choice(range(min_num_taxa, max_num_taxa + 1), size=episodes_in_run)
    total_episodes_in_batch = sum(n_taxa_list) - 3 * episodes_in_run
    batch = BatchEGAT(total_episodes_in_batch, episodes_in_parallel, max_num_taxa)
    for episode in range(episodes_in_run):
        n_taxa = n_taxa_list[episode]
        num_instances += episodes_in_parallel
        d_list = []
        for _ in range(episodes_in_parallel):

            idx = random.sample(range(dim_dataset), k=n_taxa)
            d_list.append(m[idx, :][:, idx])

        difference_mean, better, equal, variance_probs = pol.episode(d_list, n_taxa, batch)
        print(run + 1, num_instances,  "taxa ", n_taxa,
              "   difference mean", difference_mean, "   better", better,  "   equal", equal)
        print(variance_probs)

    loss = trainer.train(batch)
    print("mean loss: ", loss)
    if run > 100 and loss < best_mean_loss:
        if directory is not None and save:
            shutil.rmtree(directory)
        directory = dgn.save_net(folder, loss, params, supervised=False)
        best_mean_loss = loss