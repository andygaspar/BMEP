
'''
def initializer(self, init_steps):
adj_mats = self.initial_adj_mat(device=self.device, n_problems=1)
self.root = NodeTorch(adj_mats, step_i=3, d=self.d, n_taxa=self.n_taxa, c=self.init_c, parent=None,
                      rollout_=self.rollout_, compute_scores=self.compute_scores, device=self.device)
self.d = torch.tensor(self.d, requires_grad=False).to(self.device)
nodes_level = [self.root]

for step in range(3, init_steps):
    idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
    idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(self.device), idxs_list[1], idxs_list[2])
    repetitions = 3 + (step - 3) * 2
    adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)
    adj_mats = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)

    self.root._children = [NodeTorch(mat.unsqueeze(0), step + 1, parent=self)
                  for mat in adj_mats]

for step in range(init_steps, self.n_taxa):
    idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
    idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(self.device), idxs_list[1], idxs_list[2])
    minor_idxs = torch.tensor([j for j in range(step + 1)]
                              + [j for j in range(self.n_taxa, self.n_taxa + step - 1)]).to(self.device)

    repetitions = 3 + (step - 3) * 2

    adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

    sol = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)
    obj_vals = self.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                       self.d[:step + 1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                       step + 1)
    obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
    adj_mats = sol.unsqueeze(0).view(15, repetitions, adj_mats.shape[1],
                                     adj_mats.shape[2])[range(15), obj_vals.indices, :, :]

return obj_vals.values, adj_mats
# u = UtcSolverTorchBackTrack(np.random.uniform(size=(9, 9)), swa_policy, max_score_normalised)
# vals, mats = u.initializer()
# print("done")


'''
