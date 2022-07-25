import abc
import copy
import json
import os

from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()
        self.loss = 0

    def get_action(self, state):
        with torch.no_grad():
            state: torch.Tensor
            return self.forward(state.unsqueeze(0))

    def take_weights(self, model_network):
        self.load_state_dict(copy.deepcopy(model_network.state_dict()))

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))

    def save_weights(self, filename: str):
        torch.save(self.state_dict(), filename + '.pt')

    def save_model(self, filename: str):
        model = self.vgg16(pretrained=True)
        torch.save(model.state_dict(), filename + '.pt')

    def save_net(self, path: str, best_loss: float, net_name: str, params: dict, prefix=""):
        new_folder = path + prefix + "_" + str(int(best_loss * 1000) / 1000)
        os.mkdir(new_folder)
        params["name"] = net_name
        params["best_loss"] = best_loss
        with open(new_folder + '/params.json', 'w') as outfile:
            json.dump(params, outfile)
        torch.save(self.state_dict(), new_folder + '/weights.pt')
        return new_folder
