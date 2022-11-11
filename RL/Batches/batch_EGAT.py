import torch
from torch.utils.data import Dataset, DataLoader


class BatchEGAT(Dataset):

    def __init__(self, total_episodes_in_batch, episodes_in_parallel, max_num_taxa):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = total_episodes_in_batch * episodes_in_parallel
        self.episodes_in_parallel = episodes_in_parallel
        self.total_episodes_in_batch = total_episodes_in_batch
        self.max_dim = max_num_taxa * 2 - 2

        self.taxa_embeddings = torch.zeros((self.batch_size, self.max_dim, 5)).to(self.device)
        self.internal_embeddings = torch.zeros((self.batch_size, self.max_dim, 5)).to(self.device)
        self.message_embeddings = torch.zeros((self.batch_size, self.max_dim ** 2, 2)).to(self.device)
        self.size_mask = torch.zeros((self.batch_size, self.max_dim, 2)).to(self.device)
        self.current_mask = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.action_mask = torch.zeros((self.batch_size, self.max_dim, self.max_dim)).to(self.device)
        self.actions = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)
        self.baselines = torch.zeros(self.batch_size).to(self.device)
        self.rewards = torch.zeros(self.batch_size).to(self.device)
        self.size = self.current_mask.shape[0]
        self.index = 0

    def add_states(self, taxa_embeddings, internal_embeddings, message_embeddings, current_mask, size_mask, action_mask,
                   actions, n_taxa, m):
        self.taxa_embeddings[self.index: self.index + self.episodes_in_parallel, : m, :] = taxa_embeddings
        self.internal_embeddings[self.index: self.index + self.episodes_in_parallel, : m, :] = internal_embeddings
        self.message_embeddings[self.index: self.index + self.episodes_in_parallel, : m ** 2, :] = message_embeddings
        self.current_mask[self.index: self.index + self.episodes_in_parallel, : m, : m] = current_mask
        self.size_mask[self.index: self.index + self.episodes_in_parallel, : m, :] = size_mask
        self.action_mask[self.index: self.index + self.episodes_in_parallel, : m] = action_mask

        self.actions[self.index: self.index + self.episodes_in_parallel] = self.adjust_actions(actions, m)
        self.index += self.episodes_in_parallel

    def add_rewards_baselines(self, rewards, baselines, n_taxa):
        steps = n_taxa - 3
        self.rewards[self.index - self.episodes_in_parallel * steps: self.index] = rewards.repeat(steps)
        self.baselines[self.index - self.episodes_in_parallel * steps: self.index] = baselines.repeat(steps)

    def __getitem__(self, index):
        return self.taxa_embeddings[index], self.internal_embeddings[index], self.message_embeddings[index], \
               self.current_mask[index], self.size_mask[index], self.action_mask[index]

    def get_arb(self, index):
        return self.actions[index], self.rewards[index], self.baselines[index]

    def adjust_actions(self, actions, m):
        return torch.div(actions, m, rounding_mode='trunc') * self.max_dim + actions % m
