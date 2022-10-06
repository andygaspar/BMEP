import copy
import os
import time

import networkx as nx
import torch
import numpy as np
from Instances.instance import Instance
import random


class Generator:

    def __init__(self, name_folder, num_instances, d_mat_initial, dim_min, dim_max, max_time,
                 total_time=3_600, pardi_solver=False):
        self.name_folder = name_folder
        self.tau = None
        self.total_time = total_time
        self.max_time = max_time
        self.dim_min, self.dim_max = dim_min, dim_max
        self.dim_range = range(dim_min, dim_max + 1)
        self.m = dim_max*2 - 2
        self.d_mat_initial = d_mat_initial
        self.num_instances = num_instances
        self.pardi_solver = pardi_solver
        self.d_mats = np.zeros((self.num_instances, self.m, self.m))
        self.d_masks = np.zeros((self.num_instances, self.m, self.m))
        self.initial_masks = np.zeros((self.num_instances, self.m, 2))
        self.adj_mats = np.zeros((self.num_instances, self.m, self.m))
        self.ad_masks = np.zeros((self.num_instances, self.m, 2))
        self.masks = np.zeros((self.num_instances, self.m, self.m))
        self.y = np.zeros((self.num_instances, self.m, self.m))
        self.problem = np.zeros(self.num_instances)
        self.completed = False
        self.generate()

    def generate(self):
        random.seed(0)
        i = 0
        t = time.time()
        problem = 0
        while i < self.num_instances and time.time() - t < self.total_time:
            out_time, instance = True, None
            dim = np.random.choice(self.dim_range)
            while time.time() - t < self.total_time and out_time:
                idx = random.sample(range(self.d_mat_initial.shape[0]), k=dim)
                # print("start", dim)
                instance = Instance(self.d_mat_initial[idx, :][:, idx], max_time=self.max_time,
                                    pardi_solver=self.pardi_solver)
                out_time = instance.out_time
                # print("end", out_time)
            if instance is not None:
                j = 0
                d_mat = np.zeros((self.m, self.m))
                d_mat[:dim, :][:, :dim] = instance.d
                d_mask = copy.deepcopy(d_mat)
                d_mask[d_mask > 0] = 1

                while i < self.num_instances and j < dim - 3:
                    self.initial_masks[i, :dim, 0] = 1
                    self.initial_masks[i, dim:, 1] = 1
                    self.d_mats[i] = d_mat
                    self.d_masks[i] = d_mask
                    self.adj_mats[i][:dim*2 - 2, :dim*2 - 2] = instance.adj_mats[j]
                    a = np.sum(self.adj_mats[i], axis=-1)
                    self.ad_masks[i, :, 0] = a
                    self.masks[i][:dim*2 - 2, :dim*2 - 2] = instance.masks[j]
                    self.y[i][:dim*2 - 2, :dim*2 - 2] = instance.results[j]
                    self.problem[i] = problem
                    i += 1
                    j += 1
                    print(i)

                problem += 1

        if i == self.num_instances - 1:
            self.completed = True
        self.ad_masks[self.ad_masks > 0] = 1
        self.ad_masks[:, :, 1] = np.abs(self.ad_masks[:, :, 0] - 1)

        self.d_mats = torch.tensor(self.d_mats)
        self.d_masks = torch.tensor(self.d_masks)
        self.initial_masks = torch.tensor(self.initial_masks)
        self.adj_mats = torch.tensor(self.adj_mats)
        self.ad_masks = torch.tensor(self.ad_masks)
        self.masks = torch.tensor(self.masks)
        self.tau = self.compute_taus()
        self.y = torch.tensor(self.y)
        self.problem = torch.tensor(self.problem)
        path = 'Data_/Datasets/' + self.name_folder + '_' + str(self.dim_min) + '_' + str(self.dim_max) + '/'
        os.mkdir(path)
        torch.save(self.d_mats, path + 'd_mats.pt')
        torch.save(self.d_masks, path + 'd_masks.pt')
        torch.save(self.initial_masks, path + 'initial_masks.pt')
        torch.save(self.adj_mats, path + 'adj_mats.pt')
        torch.save(self.ad_masks, path + 'ad_masks.pt')
        torch.save(self.masks, path + 'masks.pt')
        torch.save(self.y, path + 'y.pt')
        torch.save(self.tau, path + 'taus.pt')
        torch.save(self.problem, path + 'problems.pt')

    def compute_taus(self):
        taus = []
        for adj in self.adj_mats:
            g = nx.from_numpy_matrix(adj.to("cpu").numpy())
            Tau = nx.floyd_warshall_numpy(g)
            Tau[np.isinf(Tau)] = 0
            taus.append(Tau)

        return torch.tensor(taus)

