import GPyOpt

configs = {
'path': 'Net/Nets/GNN1/',
'net_name': "GNN_1",
}

trainer = NetTrainer(configs, 'params.json')


bounds = [
          {'name': 'var_1', 'type': 'continuous', 'domain': (-8, -2)},
          {'name': 'var_2', 'type': 'discrete', 'domain': (7, 8, 9, 10)},
          ]

bopt = GPyOpt.methods.BayesianOptimization(trainer, domain=bounds, model_type='GP', acquisition_type='EI',
                                           verbosity=True, maximize=False)

bopt.get_evaluations()
bopt.plot_convergence()
bopt.run_optimization(max_iter=20, verbosity=True)
bopt.get_evaluations()
bopt.plot_convergence()
bopt.save_evaluations('net_trainer_evaluations')

