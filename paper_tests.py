"""
Stochastic approximation approach for value function approximation in Dynamic Repositioning and
Rerouting Problem (DRRP). Based on Powell, Ruszczynksi, and Topaloglu, "Learning Algorithms for
Separable Approximations of Discrete Stochastic Optimization Problems", Math of OR, 2004.
"""
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import pickle
import numpy as np
import os
from itertools import product
from drrp import SAModel

if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('vfs'):
    os.makedirs('vfs')

# Load system data
cost_params = {'lost demand cost': 1., 'vehicle movt cost': 1e-3,
               'load cost': 1e-3, 'unload cost': 1e-3,
               'lost bike cost': 20, 'created bike cost': 20,
               'lost demand cost spread low': 0.5, 'lost demand cost spread high': 1.5}
alg_params = {'n_iter': 50, 'ss_rule': 'PRT', 'max_dev': 10, '1k_const': 40, 'ss_const': 0.2,
              'relax_s1': 'All z', 'plot_every': 10, 'cost_eval_samples': 100, 'eval_cost_every': 5,
              'final_sol': True, 'final_sol_method': 'exact', 'save_iter_models': False,
              'eval_cost_k': [10, 20, 50], 'random_s1': False, 'nominal_s2': False}

T = 12  # Nmber of time steps

# Loop through saved problem instances of this size
n_v_array = {9: [1, 3], 16: [1, 5], 25: [1, 5, 9], 36: [1, 5, 11],
             64: [1, 9, 15], 100: [1, 9, 19], 225: [1, 13, 25], 400: [1, 15, 35]}
for (N, i) in product([9], range(1, 11)):  # No. of nodes, vehicles, instance #
    for V in n_v_array[N]:
        print "Testing N=%d, V=%d, T=%d, instance %d" % (N, V, T, i)
        filename = "network_data/N%03d_V%02d_T%02d" % (N, V, T) + '/instance_%02d.pkl' % i
        if os.path.exists(filename):
            with open(filename, 'rb') as input_file:
                nw = pickle.load(input_file)
            nw.dv_0 = np.zeros((V, 1))

            # Set up model
            s = SAModel(nw, cost_params, alg_params, T, i, label='_regular')

            # Run approximation algorithm and output results
            s.eval_no_action_cost()
            s.approx()
            # s.integer_only()
        else:
            print "File " + filename + " does not exist! Skipping."
