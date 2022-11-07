import abc
import copy
import json
import os

from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, normalisation_factor):
        super(Network, self).__init__()
        self.normalisation_factor = normalisation_factor
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

    def save_net(self, folder: str, score: float, params: dict, prefix="", supervised=True):
        if supervised:
            new_folder = 'Net/Nets/Supervised/' + folder + '/' + prefix + "_" + str(int(score * 1000) / 1000) + "_0"
        else:
            new_folder = 'Net/Nets/RlNets/' + folder + '/' + prefix + "_" + str(int(score * 1000) / 1000) + "_0"
        while os.path.isdir(new_folder):
            new_folder += "i"
        os.mkdir(new_folder)
        score = "best_loss" if supervised else "best mean difference"
        params["net"][score] = score
        params["net"]["normalisation factor"] = self.normalisation_factor
        with open(new_folder + '/params.json', 'w') as outfile:
            json.dump(params, outfile)
        torch.save(self.state_dict(), new_folder + '/weights.pt')
        return new_folder
