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
        #open first level branches
        _, mask = self.get_masks(adj_mat)
        idx_tensors = list(torch.tensor([x, y]) for x, y in zip(*torch.nonzero(mask, as_tuple=True)))
        branch_mats = [self.add_node(copy.deepcopy(adj_mat), idxs, 3, self.n) for idxs in idx_tensors]

        #open second level branches
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
        init_batch = adj_mat.repeat_interleave(repeats=self._n_evaluations, dim=0)
       ##### TODO ####
        acts = self._dropout_nn(init_batch)  # assuming the nn returns an action for each input
        y, _ = self.net((adj_mat, ad_mask.unsqueeze(0), self.d.unsqueeze(0),
                         d_mask.unsqueeze(0),
                         size_mask.unsqueeze(0), initial_mask.unsqueeze(0), mask.unsqueeze(0),
                         tau.unsqueeze(0),
                         tau_mask.unsqueeze(0), None))
        a_max = torch.argmax(y.squeeze(0))
        idx_tensor = torch.tensor([torch.div(a_max, self.m, rounding_mode='trunc'),
                                   a_max % self.m]).to(self.device)
        # store the results of the first actions applied to the initial instances
        init_batch = self._update_instances(init_batch, acts)

        # create a copy of the batch to use for exploration
        new_batch = copy.deepcopy(init_batch)
        for i in range(step_n, self.n):
            acts = self.net(new_batch)
            new_batch = self._update_instances(new_batch, acts)

        # update the best solution with the best found in this search round
        self._update_best_solution(new_batch)

        # select the "self._n_branches" best performing actions
        best_instances = self._select_best_instances(new_batch)
        # return the instances corresponding to the best actions
        return init_batch[best_instances]
        #################


    '''
    Apply selected actions to a batch
    '''
    def _update_instances(self, instance_batch, actions):
        pass

    '''
    Evaluate the found solutions and update the best if necessary
    '''

    def _update_best_solution(self, instance_batch):
        pass

    '''
    Evaluate the found solutions and select the best "self._n_branches" ones 
    '''

    #### TODO ####
    # Handle the following special cases:
    # 1. Actions associated with multiple solutions, pick the max
    # 2. Number of actions is less than the number of branches
    ##############
    def _select_best_instances(self, instance_batch):
        obj_vals = [self.compute_obj_val_from_adj_mat(adj_mat.to("cpu").numpy(),
                                                      self.d_mat.to("cpu").numpy(),
                                                      self.n)
                    for adj_mat in instance_batch]

        return torch.argsort(obj_vals)[:self._n_branches]
