import torch
import copy

class SearchSolver:

    def __int__(self, n_branches, n_evaluations, obj_fun, dropout_nn):
        self._n_branches = n_branches
        self._n_evaluations = n_evaluations
        self._dropout_nn = dropout_nn
        self._obj_fun = obj_fun #assume the obj_fun is a vectorized fun working with torch.Tensors

        self._best_solution = None

    '''
    Solve a batch of instances 
    '''
    def solve(self, instance):

        instance_batch = self._open_branches(instance)

        done = False
        while not done:
            instance_batch = self._solve_step(instance_batch)
            done = self._check_reached_leaves(instance_batch)

        if self._best_solution is not None:
            return self._best_solution
        else:
            raise Exception()

    '''
    Open as many branches as possible and return them as a batch of instances
    '''
    def _open_branches(self, instance):
        pass

    '''
    Given a batch of instances, search solutions from there on and pick the best "self._n_branches" actions.
    ** HERE WE ASSUME FOR SIMPLICITY THAT INSTANCE BATCH IS A MATRIX WITH DIMENSIONS (n_batch, input_dim) **
    '''
    def _solve_step(self, instance_batch):
        init_batch = instance_batch.repeat_interleave(repeats=self._n_evaluations, dim=0)
        acts = self._dropout_nn(init_batch) #assuming the nn returns an action for each input
        #store the results of the first actions applied to the initial instances
        init_batch = self._update_instances(init_batch, acts)

        #create a copy of the batch to use for exploration
        new_batch = copy.deepcopy(init_batch)
        done = self._check_reached_leaves(new_batch)
        while not done:
            acts = self._dropout_nn(new_batch)
            new_batch = self._update_instances(new_batch, acts)

            done = self._check_reached_leaves(new_batch)

        #update the best solution with the best found in this search round
        self._update_best_solution(new_batch)

        #select the "self._n_branches" best performing actions
        best_instances = self._select_best_instances(new_batch)
        #return the instances corresponding to the best actions
        return init_batch[best_instances]

    '''
    Check if the solver reached the leaves of the search tree
    '''
    def _check_reached_leaves(self, instance_batch):
        pass

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
    def _select_best_instances(self, instance_batch):
        return torch.argsort(self._obj_fun(instance_batch))[:self._n_branches]