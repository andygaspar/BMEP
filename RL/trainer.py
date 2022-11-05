import copy
import itertools

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Batch(Dataset):

    def __init__(self, total_episodes_in_batch, episodes_in_parallel, max_num_taxa):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = total_episodes_in_batch * episodes_in_parallel
        self.episodes_in_parallel = episodes_in_parallel
        self.total_episodes_in_batch = total_episodes_in_batch
        self.max_dim = max_num_taxa * 2 - 2

        self.d_mats = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.d_masks = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.initial_masks = torch.zeros((self.batch_size, self.max_dim, 2)).to(self.device)
        self.adj_mats = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.ad_masks = torch.zeros((self.batch_size, self.max_dim, 2)).to(self.device)
        self.masks = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.tau = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.tau_mask = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.size_masks = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.actions = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)
        self.baselines = torch.zeros(self.batch_size).to(self.device)
        self.rewards = torch.zeros(self.batch_size).to(self.device)
        self.size = self.d_mats.shape[0]
        self.index = 0

    def add_states(self, d_mats, d_masks, initial_masks, adj_mats, ad_masks, masks, tau, tau_mask, size_masks,
                   actions, m):
        self.d_mats[self.index: self.index + self.episodes_in_parallel, : m, : m] = d_mats
        self.d_masks[self.index: self.index + self.episodes_in_parallel, : m, : m] = d_masks
        self.initial_masks[self.index: self.index + self.episodes_in_parallel, : m] = initial_masks
        self.adj_mats[self.index: self.index + self.episodes_in_parallel, : m, : m] = adj_mats
        self.ad_masks[self.index: self.index + self.episodes_in_parallel, : m] = ad_masks
        self.masks[self.index: self.index + self.episodes_in_parallel, : m, : m] = masks
        self.tau[self.index: self.index + self.episodes_in_parallel, : m, : m] = tau
        self.tau_mask[self.index: self.index + self.episodes_in_parallel, : m, : m] = tau_mask
        self.size_masks[self.index: self.index + self.episodes_in_parallel, : m, : m] = size_masks
        self.actions[self.index: self.index + self.episodes_in_parallel] = self.adjust_actions(actions, m)
        self.index += self.episodes_in_parallel

    def add_rewards_baselines(self, rewards, baselines, n_taxa):
        steps = n_taxa - 3
        self.rewards[self.index - self.episodes_in_parallel*steps: self.index] = rewards.repeat(steps)
        self.baselines[self.index - self.episodes_in_parallel * steps: self.index] = baselines.repeat(steps)

    def __getitem__(self, index):
        return self.adj_mats[index], self.ad_masks[index], self.d_mats[index], self.d_masks[index],\
               self.size_masks[index], self.initial_masks[index],  self.masks[index], self.tau[index], \
               self.tau_mask[index]

    def get_arb(self, index):
        return self.actions[index], self.rewards[index], self.baselines[index]

    def adjust_actions(self, actions, m):
        return torch.div(actions, m, rounding_mode='trunc') * self.max_dim + actions % m


class Trainer:

    def __init__(self, net, optimiser, training_batch_size, training_loops):
        self.net = net
        self.optimiser = optimiser
        self.training_batch_size = training_batch_size
        self.loops = training_loops
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, batch: Batch):
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
        loss = torch.sum(l_probs * (baseline - obj_vals) / baseline)
        loss.backward()
        self.optimiser.step()
        return loss.detach().item()
