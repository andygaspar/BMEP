import copy
import itertools

import networkx as nx
import numpy as np
import torch


class Trainer:

    def __init__(self, net, optimiser, training_batch_size, training_loops):
        self.net = net
        self.optimiser = optimiser
        self.training_batch_size = training_batch_size
        self.loops = training_loops
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, batch):
        random_idx = torch.randperm(batch.batch_size).to(self.device)
        loss = 0
        for _ in range(self.loops):
            for i in range(batch.batch_size//self.training_batch_size):
                idxs = random_idx[i * self.training_batch_size: (i + 1) * self.training_batch_size]
                loss += self.train_mini_batch(idxs, batch)

            residual_dataset = batch.batch_size % self.training_batch_size
            if residual_dataset > 0:
                idxs = random_idx[-batch.batch_size % self.training_batch_size:]
                loss += self.train_mini_batch(idxs, batch)
        return loss

    def train_mini_batch(self, idxs, batch):
        actions, obj_vals, baseline = batch.get_arb(idxs)
        state = batch[idxs]
        self.optimiser.zero_grad()
        probs, l_probs = self.net(state)
        l_probs = l_probs[(torch.arange(0, len(l_probs), dtype=torch.long), actions)]
        # pb = torch.distributions.Categorical(probs)
        loss = torch.mean(l_probs * (obj_vals - baseline) / baseline)*10
        loss.backward()
        self.optimiser.step()
        return loss.detach().item()

    def train_acr(self, batch):
        pass