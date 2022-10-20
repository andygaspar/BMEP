import networkx as nx
import numpy as np
import torch
import itertools
import copy

from Solvers.NetSolvers.net_solver import NetSolver


class SearchSolver(NetSolver):

    def __int__(self, n_branches, n_evaluations, dropout_nn, d_mat):
        super(SearchSolver, self).__init__(d_mat, dropout_nn)
        self._n_branches = n_branches
        self._n_evaluations = n_evaluations
        self.d_mat = d_mat

        self._best_solution = None
        self._best_obj_val = float('inf')

    '''
    Solve a batch of instances 
    '''

    def solve(self):

        adj_mat, size_mask, initial_mask, d_mask = self.initial_mats()

        adj_mat = self._open_branches(adj_mat)
        current_branches = adj_mat.shape[0]

        for i in range(5, self.n):
            adj_mat = self._solve_step(i, (adj_mat, size_mask, initial_mask, d_mask))

        if self._best_solution is not None:
            return self._best_solution
        else:
            raise Exception()

    '''
    Open second and third level branches given the initial instance adjacency matrix
    '''

    def _open_branches(self, adj_mat):
        # open first level branches
        _, mask = self.get_masks(adj_mat)
        idx_tensors = list(torch.tensor([x, y]) for x, y in zip(*torch.nonzero(mask, as_tuple=True)))
        branch_mats = [self.add_node(copy.deepcopy(adj_mat), idxs, 3, self.n) for idxs in idx_tensors]

        # open second level branches
        masks = [self.get_masks(b_mat)[1] for b_mat in branch_mats]
        idx_tensors = [list(torch.tensor([x, y]) for x, y in zip(*torch.nonzero(m, as_tuple=True))) for m in masks]
        branch_mats = [self.add_node(copy.deepcopy(b_mat), idxs, 4, self.n).unsqueeze(0)
                       for b_mat in branch_mats for idxs in idx_tensors]

        return torch.cat(itertools.chain.from_iterable(branch_mats), dim=0)

    '''
    Given a batch of instances, search solutions from there on and pick the best "self._n_branches" actions.
    ** HERE WE ASSUME FOR SIMPLICITY THAT INSTANCE BATCH IS A MATRIX WITH DIMENSIONS (n_batch, input_dim) **
    '''
    def _solve_step(self, step_n, instance_batch):
        adj_mat, size_mask, initial_mask, d_mask = instance_batch

        adj_mat, ad_mask, mask, tau, tau_mask = self._initial_mats(step_n, adj_mat)
        adj_mat, ad_mask, mask, tau, tau_mask = self.__sample_actions(step_n, adj_mat, ad_mask, d_mask, size_mask, initial_mask, mask, tau, tau_mask)

        # store the results of the first actions applied to the initial instances
        new_adj_mats = copy.deepcopy(adj_mat)

        for i in range(step_n + 1, self.n):
            adj_mat, ad_mask, mask, tau, tau_mask = self.__sample_actions(i, adj_mat, ad_mask, d_mask, size_mask,
                                                                          initial_mask, mask, tau, tau_mask)

        # update the best solution with the best found in this search round
        instance_values = self._evaluate_solutions(adj_mat)
        self._update_best_solution(new_adj_mats, instance_values)

        # select the "self._n_branches" best performing actions and return them
        return self._select_best_instances(new_adj_mats, instance_values)

    def _initial_mats(self, step_n, adj_mat):
        adj_mat = adj_mat.repeat_interleave(repeats=self._n_evaluations, dim=0)
        ad_mask = self.__get_ad_mask(step_n)
        mask = torch.triu(adj_mat)
        tau, tau_mask = self.__get_tau_mats(adj_mat)

        return adj_mat, ad_mask, mask, tau, tau_mask

    '''
    Sample an action and apply it to the instances matrices
    '''
    def __sample_actions(self, step_n, adj_mat, ad_mask, d_mask, size_mask, initial_mask, mask, tau, tau_mask):
        y, _ = self.net((adj_mat, ad_mask, self.d.unsqueeze(0),
                         d_mask.unsqueeze(0),
                         size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask,
                         tau,
                         tau_mask, None))
        a_max = torch.argmax(y, dim=-1, keepdim=True)
        acts = torch.hstack([torch.div(a_max, self.m, rounding_mode='trunc'), a_max % self.m]).to(self.device)

        return self._update_instances(step_n, acts, adj_mat, ad_mask)

    '''
    Generate ad_mask exploiting the fact that, when at step i, all taxon up to i are in the tree
    '''
    def __get_ad_mask(self, step_n):
        ad_mask = torch.zeros((1, self.d.shape[0], 2)).to(self.device)
        ad_mask[:, :step_n, 0] = 1
        ad_mask[:, step_n:, 1] = 1
        return ad_mask

    def __get_tau_mats(self, adj_mats):
        gs = [nx.from_numpy_matrix(adj_mat.to('cpu').numpy()) for adj_mat in adj_mats]
        tau = np.concatenate([nx.floyd_warshall_numpy(g)[np.newaxis, :, :] for g in gs], axis=0)
        tau[np.isinf(tau)] = 0
        tau = torch.tensor(tau).to(torch.float).to(self.device)
        tau_mask = copy.deepcopy(tau)
        tau_mask[tau_mask > 0] = 1
        return tau, tau_mask


    '''
    Apply selected actions to a batch
    '''

    def _update_instances(self, step_n, acts, adj_mat, ad_mask):
        #update ad_mask
        ad_mask[:, step_n] = torch.tensor([0, 1]).view((1, 1, -1))
        #update adjacency matrix
        adj_mat[:, acts[:, 0], acts[1]] = adj_mat[:, acts[:, 1], acts[:, 0]] = 0  # detach selected
        adj_mat[:, acts[:, 0], self.n + step_n - 2] = adj_mat[:, self.n + step_n - 2, acts[:, 0]] = 1  # reattach selected to new
        adj_mat[:, acts[:, 1], self.n + step_n - 2] = adj_mat[:, self.n + step_n - 2, acts[:, 1]] = 1  # reattach selected to new
        adj_mat[:, step_n, self.n + step_n - 2] = adj_mat[:, self.n + step_n - 2, step_n] = 1  # attach new
        #update mask
        mask = torch.triu(adj_mat)
        #update tau and tau_mask
        tau, tau_mask = self.__get_tau_mats(adj_mat)

        return adj_mat, ad_mask, mask, tau, tau_mask

    '''
    Evaluate the found solutions and update the best if necessary
    '''

    def _update_best_solution(self, instance_batch, instance_values):
        best_val, best_id = np.amax(instance_values), np.argmax(instance_values)
        if best_val < self._best_obj_val:
            self._best_solution = instance_batch[best_id]
            self._best_obj_val = best_val

    '''
    Evaluate the found solutions and select the best "self._n_branches" ones 
    '''

    #### TODO ####
    # Handle the following special cases:
    # 1. Actions associated with multiple solutions, pick the max
    # 2. Number of actions is less than the number of branches
    ##############
    def _select_best_instances(self, instance_batch, instance_values):
        best_idxs = np.argsort(instance_values)[:self._n_branches]
        return instance_batch[best_idxs]

    def _evaluate_solutions(self, adj_mats):
        return np.array([self.compute_obj_val_from_adj_mat(adj_mat.to("cpu").numpy(), self.d.to("cpu").numpy(), self.n) for adj_mat in adj_mats])
