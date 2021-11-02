import argparse
import scipy.io
import numpy as np
import elfi
import torch
import time

from elfi.examples.LGSSM import LGSSM
from elfi.examples.stochastic_volatility import StochasticVolatility
from elfi.examples.dynamic_toy_model import DynamicToyProcess
from elfi.examples.umap_tasks import UMAPTasks
from elfi.examples.gaze_selection import GazeSelection

import warnings
warnings.filterwarnings('ignore')

def samples_to_numpy(samples, param_names):
    sample_size = len(samples[0][param_names[0]])
    return np.vstack([[np.array([[sample[name][i] for name in param_names] for sample in samples])] for i in range(sample_size)])

def rmse(samples, param_path):
    param_path = np.array(param_path)
    return np.sqrt(np.mean(np.sum((samples - param_path)**2, axis=2), axis=0))

def result_dict(time, samples, estimates, param_names, param_path=None, discrepancies=[], predictions=None, trajectories=None):
    mdict = {}
    S = samples_to_numpy(samples, param_names)
    for i, name in enumerate(param_names):
        mdict[name] = S[:, :, i].T

    if param_path:
        mdict['true_params'] = np.array([param_path])
    mdict['estimates'] =  np.array([estimates])
        
    if len(discrepancies) > 0:
        mdict['discrepancies'] = discrepancies
    if predictions:
        pred = samples_to_numpy(predictions, param_names)
        for i, name in enumerate(param_names):
            mdict[name + '_pred'] = pred[:, :, i].T
    if trajectories:
        traj = samples_to_numpy(trajectories, param_names)
        for i, name in enumerate(param_names):
            mdict[name + '_traj'] = traj[:, :, i].T

    mdict['time'] = time
    return mdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim')
    parser.add_argument('--meth')
    parser.add_argument('--seed')
    parser.add_argument('--budget')
    parser.add_argument('--tasks')
    
    args = parser.parse_args()
    start = time.time()

    # set experiment parameters
    steps = 50
    num_latents = 3
    sample_size = 1000
    init_evidence = 20
    
    num_acq_points = int(args.budget) if args.budget else 2 # 2 10 # 6
    num_tasks = int(args.tasks) if args.tasks else 2 # 2 10 # 6
    
    # fix random seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # setup the simulator
    sim = str(args.sim)
    n_obs = 10
    if sim == 'lgssm':
        x = 100.0
        process = LGSSM(true_params=[x], n_obs=n_obs, seed=seed)
    elif sim == 'toy':
        t1 = 0.0
        process = DynamicToyProcess(true_params=[t1], n_obs=n_obs, seed=seed)
    elif sim == 'sv':
        mu = 0.0
        beta = 0.0
        v0 = 1.0
        process = StochasticVolatility(true_params=[mu, beta, v0], n_obs=n_obs, seed=seed)
    elif sim == 'umap':
        process = UMAPTasks(n_obs=n_obs, seed=seed)
    elif sim == 'gaze':
        eye_latency = 37
        ocular_noise = 0.01
        spatial_noise = 0.09
        process = GazeSelection(epochs=1e5, true_params=[eye_latency, ocular_noise, spatial_noise], n_obs=n_obs, seed=seed)

    # setup the inference method
    meth = str(args.meth)
    if meth == 'bnn' or meth == 'qehvi' or meth == 'blr':
        steps -= 1
        tkwargs = {'dtype': torch.float, 'device': torch.device('cpu')}
        method = elfi.LMCInference(process=process, num_tasks=num_tasks, num_latents=num_latents, 
                               num_inducing_points=50, learn_inducing_locations=True, tkwargs=tkwargs)
        method.fit(steps, num_initial_evidence=init_evidence , num_acquisition_points=num_acq_points,
                   acq_method=meth, lmc_epochs=2000, lmc_posterior_samples=sample_size,
                   prediction_samples=sample_size)
        pred = method.predictions
        traj = method.sampled_trajectories
    elif meth == 'bolfi':
        method = elfi.SequentialBOLFI(process=process)
        method.fit(steps, num_initial_evidence=init_evidence, num_acquisition_points=num_acq_points, 
                    sample_size=sample_size, seed=seed)
        pred = None
        traj = None
    elif meth == 'SNPE' or meth == 'SNLE' or meth == 'SNRE':
        discr = True if sim == 'umap' else False
        budget = init_evidence + steps * num_acq_points
        method = elfi.SequentialNDE(process=process, discr=discr)
        method.fit(steps, meth, budget=budget, num_posterior_samples=sample_size)
        pred = None
        traj = None

    # umap simulator does not have true parameters
    param_path = None if sim == 'umap' else process.param_path

    end_time = (time.time() - start) / 60.

    from pathlib import Path
    Path(sim + '/' + meth + '-w' + str(num_tasks) + '-s' + str(num_acq_points) ).mkdir(parents=True, exist_ok=True)
    save_file_name = sim + '/' + meth + '-w' + str(num_tasks) + '-s' + str(num_acq_points)  + '/rmse-' + str(seed) + '.mat'
    mdict = result_dict(end_time, method.samples, method.estimates, process.param_names, param_path, 
                    predictions=pred, trajectories=traj)
    scipy.io.savemat(save_file_name, mdict=mdict)
