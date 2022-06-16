import torch

from Net.gnn import DGN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d_mats = torch.load("Data_/Dataset/d_mats.pt").to(torch.float).to(device)
initial_masks = torch.load("Data_/Dataset/initial_masks.pt").to(torch.float).to(device)
adj_mats = torch.load("Data_/Dataset/adj_mats.pt").to(torch.float).to(device)
masks = torch.load("Data_/Dataset/masks.pt").to(torch.float).to(device)
y = torch.load("Data_/Dataset/y.pt").to(torch.float).to(device)

dgn = DGN(4, 64, 64, 7)
y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
                    masks[0].unsqueeze(0))

from Pardi.pardi import Pardi
import numpy as np

T = np.array([[0, 3, 5, 5, 2, 4],
              [3, 0, 3, 4, 4, 3],
              [5, 3, 0, 2, 5, 4],
              [5, 4, 2, 0, 5, 3],
              [2, 4, 5, 5, 0, 3],
              [4, 3, 4, 3, 3, 0]])
p = Pardi(T)

d = np.array([[0., 0.0093179, 0.015149, 0.0150827, 0.0187477, 0.0310693],
              [0.0093179, 0., 0.0142182, 0.016167, 0.0207826, 0.0326569],
              [0.015149, 0.0142182, 0., 0.0135819, 0.0246194, 0.0361711],
              [0.0150827, 0.016167, 0.0135819, 0., 0.0252762, 0.033975],
              [0.0187477, 0.0207826, 0.0246194, 0.0252762, 0., 0.0402366],
              [0.0310693, 0.0326569, 0.0361711, 0.033975, 0.0402366, 0.]])

from Instances.ip_solver import solve

t = solve(d, 600)

T = np.array([[0, 3, 5, 5, 2, 4],
              [3, 0, 3, 4, 4, 3],
              [5, 3, 0, 2, 5, 4],
              [5, 4, 2, 0, 5, 3],
              [2, 4, 5, 5, 0, 3],
              [4, 3, 4, 3, 3, 0]])
p = Pardi(T)
