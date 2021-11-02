Likelihood-Free Inference in State-Space Models with Unknown Dynamics
=====================================================================

This package contains the codes required to run the experiments in the paper. The simulators used for the State-Space Models in the experiments are implemented based on Engine for Likelihood-free Inference (ELFI) models.


Installation
------------
We recommend using an Anaconda environment. To create and activate the conda environment with all dependencies installed, run:

```
conda create -c conda-forge --name env --file lfi-requirements.txt
conda activate env
pip install -e .
pip install sbi blitz-bayesian-pytorch stable_baselines3
```

For the GP-SSM and PR-SSM methods, we recommend creating a separate environment, in which one should install tensorflow, and then clone the 'custom_multiouput' branch of the GPflow from https://github.com/ialong/GPflow. Once GPflow is installed, one should clone GPt from https://github.com/ialong/GPt and execute 'experiments/run_gpssms.py', the code will complete 30 repletions of experiments with tractable likelihoods.

Running the experiments
-----------------------
The experiment scripts can be found in the 'experiments/' folder. To run the experiments on one of the considered SSM, one should run the 'run_experiment.py' script with the following arguments (options are in the parentheses): --sim ('lgssm', 'toy', 'sv', 'umap', 'gaze'), --meth ('bnn', 'qehvi', 'blr', 'SNPE', 'SNLE', 'SNRE'), --seed (any seed number), --budget (available simulation budget for each new state), --tasks (number of tasks considered/ moving window size for LMC-BNN, LMC-qEHVI and LMC-BLR methods). For instance:

```
python3 experiments/run_experiment.py --sim=lgssm --meth=bolfi --seed=0 --budget=2 --tasks=2
```

The results will be saved in the corresponding folders 'experiments/[sim]/[meth]-w[tasks]-s[budget]/'. To build plots and output the results, one should run 'collect_plots.py' script with specified arguments: --type ('inf' in case of evaluating state inference quality or 'traj' in case of evaluating the generated trajectories), --tasks (the number of tasks used by the methods). For example:

```
python3 experiments/collect_results.py --type=inf --tasks=2
```

The plots with experiment results will be stored in 'experiments/plots'.


Implementing custom simulators
------------------------------
The simulators for all experiments can be found in elfi/examples. Example implementations used in the paper are found in gaze_selection.py, umap_tasks.py, LGSSM.py (LG), dynamic_toy_model.py (NN), and stochastic_volatility.py (SV). To create a new SSM, implement a new class that inherits from elfi.DynamicProcess with custom generating function for observations, create_model(), and update_dynamic().

The code for all methods can be found in 'elfi/methods/dynamic_parameter_inference.py' and 'elfi/methods/bo/mogp.py'.

Citation
--------

```

```
