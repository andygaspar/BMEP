import random
from ast import literal_eval

import numpy as np
import pandas as pd

from Data_.data_loader import DistanceData
from Solvers.FastME.fast_me import FastMeSolver
from Solvers.RandomNni.random_nni import RandomNni

distances = DistanceData()
distances.print_dataset_names()

data_set = distances.get_dataset(3)

df = pd.read_csv('Data_/ImprovableTest/test_instances_idxs.csv')
df['idxs_list'] = df.idxs.apply(lambda t: literal_eval(t))



seed = 0
random.seed(seed)
np.random.seed(seed)
iterations = 1000

best_iter, counter, obj, seeds, fast_sol, runs = [], [], [], [], [], []

for i in range(10):
    d = data_set.get_from_idxs(df.idxs_list.iloc[i])

    fast = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True,
                        triangular_inequality=False, logs=False)
    fast.solve()
    for seed in range(100):
        print(i, seed)
        comparison = RandomNni(d, parallel=False)
        comparison.solve_sequential(iterations, fast.obj_val, fast.solution)

        best_iter.append(comparison.best_iteration)
        seeds.append(seed)

        counter.append(comparison.counter)
        obj.append(comparison.obj_val)

        runs.append(i)
        fast_sol.append(fast.obj_val)

results = pd.DataFrame({'runs': runs, 'best_iter': best_iter, 'counter': counter, 'obj': obj, 'seed': seeds, 'fast': fast_sol})
results.to_csv('random_analysis.csv', index_label=False, index=False)

import pandas as pd

df = pd.read_csv('../../DataAnalysis/random_analysis.csv')

for i in range(10):
    print( df[df.runs==i].best_iter.mean(), df[df.runs==i].best_iter.std())