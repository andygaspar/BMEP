import GPyOpt

from Net.Nets.GNN1.gnn_1 import GNN_1
from ParamTuning.net_trainer import NetTrainer

configs = {
    'path': 'Net/Nets/GNN1/',
    'net_name': "GNN_1",
}
net = GNN_1

trainer = NetTrainer(net, configs, 'params.json')

bounds = [
    {'name': 'var_1', 'type': 'continuous', 'domain': (-6, -3)},
    {'name': 'var_2', 'type': 'continuous', 'domain': (-6, -3)},
]

bopt = GPyOpt.methods.BayesianOptimization(trainer, domain=bounds, model_type='GP', acquisition_type='EI',
                                           verbosity=True, maximize=False)

# bopt.get_evaluations()
# bopt.plot_convergence()
bopt.run_optimization(max_iter=20, verbosity=True)
bopt.get_evaluations()
bopt.plot_convergence()
bopt.save_evaluations('net_trainer_evaluations')
