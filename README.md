Likelihood-Free Inference in State-Space Models with Unknown Dynamics
=====================================================================

This package contains the codes required to run the experiments in the paper. The simulators used for the State-Space Models in the experiments are implemented based on Engine for Likelihood-free Inference (ELFI) models.

# Installation
------------
We recommend setting up an Anaconda environment for a seamless experience.

#### Standard Environment Setup:

1. Create and activate a conda environment with the required dependencies:
```bash
conda create -c conda-forge --name env --file lfi-requirements.txt
conda activate env
```
2. Install the necessary packages:
```bash
pip install -e .
pip install sbi blitz-bayesian-pytorch stable_baselines3 statsmodels
```

#### Setup for GP-SSM and PR-SSM Methods:

1. First, install tensorflow in your environment.
2. Clone the 'custom_multiouput' branch of GPflow from here: https://github.com/ialong/GPflow 
3. Once GPflow is set up, clone GPt from this repository: https://github.com/ialong/GPt
4. Run the following to execute the experiments:
```bash
experiments/run_gpssms.py
```

This will execute 30 repetitions of experiments with tractable likelihoods.

#### Troubleshooting:
In the event of any issues, consider downgrading to the following package versions:
- arviz==0.11.0
- pymc3==3.10.0
- networkx==1.11


# Overview
--------

This documentation outlines the primary experiment files and their respective functionalities. They are organized into three main categories:

1. **Main Experiment Files**: These are your go-to files for running experiments, generating plots, and testing BNN transition dynamics.
2. **Simulator Experiments**: Delve into different simulators, from linear Gaussian models to eye movement control simulations for gaze-based selection.
3. **Methods**: This section showcases various methodologies, including our proposed Multi-objective LFI, BOLFI, Sequential Neural Estimators, and more. Each method's source code is located within its designated class in the specified file.

For detailed execution, navigate to the provided file paths.

**Main experiment files**:
- Run methods and experiments: experiments/run_experiment.py 
- Generate plots: experiments/collect_plots.py
- Test BNN transition dynamics/ACF plots: experiments/dynamics_test.py

**Simulator experiments**:
- Linear Gaussian: elfi/examples/LGSSM.py
- Non-linear non-Gaussian: elfi/examples/dynamic_toy_model.py
- Stochastic volatility:  elfi/examples/stochastic_volatility.py
- UMAP parameterization: elfi/examples/umap_tasks.py
- Eye movement control for gaze-based selection: elfi/examples/gaze_selection.py

**Methods**:
- Multi-objective LFI with transition model (proposed method): elfi/methods/dynamic\_parameter\_inference.py (within the LMCInference class)
- BOLFI: elfi/methods/dynamic\_parameter\_inference.py (within the SequentialBOLFI class)
- Sequential neural estimators: elfi/methods/dynamic\_parameter\_inference.py (within the SquentialNDE class)
- GP-SSM, PR-SSM: /experiments/run_gpssms.py (also runs experiments for these methods)


# Running the experiments
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


# Implementing custom simulators
------------------------------
The simulators for all experiments can be found in elfi/examples. Example implementations used in the paper are found in gaze_selection.py, umap_tasks.py, LGSSM.py (LG), dynamic_toy_model.py (NN), and stochastic_volatility.py (SV). To create a new SSM, implement a new class that inherits from elfi.DynamicProcess with custom generating function for observations, create_model(), and update_dynamic().

The code for all methods can be found in 'elfi/methods/dynamic_parameter_inference.py' and 'elfi/methods/bo/mogp.py'.

# Citation
--------

```
@article{aushev2021likelihood,
  title={Likelihood-Free Inference in State-Space Models with Unknown Dynamics},
  author={Aushev, Alexander and Tran, Thong and Pesonen, Henri and Howes, Andrew and Kaski, Samuel},
  journal={arXiv preprint arXiv:2111.01555},
  year={2021}
}

```
the arxiv version of the paper: https://arxiv.org/abs/2111.01555

