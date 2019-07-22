### About

This code was developed to produce the results in the paper "Two stage stochastic approximation for dynamic rebalancing of shared mobility systems", Transportation Research Part C: Emerging Technologies, vol. 104, pp. 110-134, July 2019. Available from the publisher Elsevier at <https://www.sciencedirect.com/science/article/pii/S0968090X18314104> or on ArXiv: <https://arxiv.org/abs/1810.01804>

### Requirements

The code is written in Python 2.7 and requires the solver Gurobi (free for academic use). Note that environment variables need to be set correctly for the solver to be found. In the case of Mac OS, which was used during development, `GRB_LICENSE_FILE` needs to point to the full path including filename of the Gurobi license file. Windows and Linux will have a similar requirement.

## Usage

The paper's results were generated with a sequence of Python scripts as follows:

### Creating synthetic networks and demand

The synthetic test networks and associated demand, on which the approximation algorithms were tested in the paper, can be created using the command 

```
python create_many_networks.py
```

at the command line. This creates problems sets for the desired network sizes `N` and number of vehicles `V` associated with each `N`, and stores the data in .pkl (Pickle) format in the subdirectories `network_data/Nnnn_Vvv_T12/`, where `nnn` is the number of network nodes, `vv` is the number of repositioning vehicles (RVs), and `T12` refers to the time horizon, which is 12 time steps by default. This directory is created automatically if it does not already exist, and will be filled with files of the form `instance_ii.pkl`, where `ii` is the ID of the random demand instance created.

### Stochastic approximation routine

The approximation routine is executed via the command

```
python paper_tests.py
```

which loops through the synthetic network data saved and generates value functions for load/unload actions, and associated optimized actions for the RVs.

The output is a set of CSV files in the folder `output/` which gets created if not already present. The filenames have a naming convention analogous to that of the synthetic networks described above, except it has an arbitrary suffix just before the extension, e.g. `_regular`, which is set when instantiating objects of the class `SAModel`, and allows the results to be processed according to the parameter set that was used for those tests. One file per problem instance `i` is created. 

### Processing results

The output files generated in multiple random problem instances are summarised by the command

```
python paper_results.py
```

which collects together the `output/*.csv` files with the same `N`, `V`, and suffix label, and measures statistics across the multiple instances `i`. This generates two more files, `output/iter_times<suffix>.csv` and `output/mean_std_stats<suffix>.csv`. 

An additional script, `paper_convert_to_latex.py`, converts the output into _LaTeX_ format for publishing purposes. It is included for completeness.

## User-defined parameters

Alongside the network and demand data, the solution algorithm requires two more dictionaries of parameters to be set. These are passed together with the network data `nw` when the stochastic approximation model is instantiated:

```
s = SAModel(nw, cost_params, alg_params, T, i, label='_regular')
```

Most of the parameters described below are self-explanatory but may be described in more detail in a later version of this document.

### Cost function parameters

The synthetic networks and demand are created without regard to any model of the users' or system operator's cost functions. You can set the cost function of the problem with a dictionary of the form

```
cost_params = {'lost demand cost': 1., 'vehicle movt cost': 1e-3,
               'load cost': 1e-3, 'unload cost': 1e-3,
               'lost bike cost': 20, 'created bike cost': 20,
               'lost demand cost spread low': 0.5, 'lost demand cost spread high': 1.5}              
```

### Algorithm parameters

The algorithm is adapted from the Separable, Projective Approximation Routine (Powell et al., 2004) referenced in the paper. It accepts a parameter dictionary of the form

```
alg_params = {'n_iter': 50, 'ss_rule': 'PRT', 'max_dev': 10, '1k_const': 40, 'ss_const': 0.2,
              'relax_s1': 'All z', 'plot_every': 10, 'cost_eval_samples': 100, 'eval_cost_every': 5,
              'final_sol': True, 'final_sol_method': 'exact', 'save_iter_models': False,
              'eval_cost_k': [10, 20, 50], 'random_s1': False, 'nominal_s2': False}
```