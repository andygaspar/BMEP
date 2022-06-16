import time

import torch
import numpy as np
from Instances.instance import Instance
import random


class Generator:

    def __init__(self, num_instances, d_mat_initial, dim, max_time, total_time=3_600):
        self.total_time = total_time
        self.max_time = max_time
        self.dim = dim
        self.m = self.dim*2 - 2
        self.d_mat_initial = d_mat_initial
        self.num_instances = num_instances
        self.d_mats = np.zeros((self.num_instances, self.m, self.m))
        self.initial_masks = np.zeros((self.num_instances, self.m, 2))
        self.adj_mats = np.zeros((self.num_instances, self.m, self.m))
        self.masks = np.zeros((self.num_instances, self.m, self.m))
        self.y = np.zeros((self.num_instances, self.m, self.m))
        self.completed = False
        self.generate()

    def generate(self):
        random.seed(0)
        i = 0
        t = time.time()
        while i < self.num_instances and time.time() - t < self.total_time:
            out_time, instance = True, None
            while time.time() - t < self.total_time and out_time:
                idx = random.sample(range(self.d_mat_initial.shape[0]), k=self.dim)
                print("start")
                if i == 51:
                    print("here")
                instance = Instance(self.d_mat_initial[idx, :][:, idx], max_time=self.max_time)
                out_time = instance.out_time
                print("end", out_time)
            if instance is not None:
                j = 0
                d_mat = np.zeros((self.m, self.m))
                d_mat[:self.dim, :][:, :self.dim] = instance.d

                while i < self.num_instances and j < self.dim - 3:
                    self.initial_masks[i, :self.dim, 0] = 1
                    self.initial_masks[i, self.dim:, 1] = 1
                    self.d_mats[i] = d_mat
                    self.adj_mats[i] = instance.adj_mats[j]
                    self.masks[i] = instance.masks[j]
                    self.y[i] = instance.results[j]
                    i += 1
                    j += 1
                    print(i)

        if i == self.num_instances - 1:
            self.completed = True

        self.d_mats = torch.tensor(self.d_mats)
        self.initial_masks = torch.tensor(self.initial_masks)
        self.adj_mats = torch.tensor(self.adj_mats)
        self.masks = torch.tensor(self.masks)
        self.y = torch.tensor(self.y)
        torch.save(self.d_mats, "Data_/Dataset/d_mats.pt")
        torch.save(self.initial_masks, "Data_/Dataset/initial_masks.pt")
        torch.save(self.adj_mats, "Data_/Dataset/adj_mats.pt")
        torch.save(self.masks, "Data_/Dataset/masks.pt")
        torch.save(self.y, "Data_/Dataset/y.pt")




"""

xv

se xv[i,j, 0] = 1
[t, Tji]

"""