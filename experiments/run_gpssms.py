import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gpflow as gp
from gpflow import settings as gps
from GPt import GPSSM_VCDT, PRSSM
from GPt.emissions import GaussianEmissions, VolatilityEmissions

from elfi.examples.LGSSM import *
from elfi.examples.stochastic_volatility import *
from elfi.examples.dynamic_toy_model import *
import time

def samples_to_numpy(samples, param_names):
    sample_size = len(samples[0][param_names[0]])
    return np.vstack([[np.array([[sample[name][i] for name in param_names] for sample in samples])] for i in range(sample_size)])


parser = argparse.ArgumentParser()
parser.add_argument('--seed')
args = parser.parse_args()

seed = args.seed if args.seed else 0
steps = 50
sample_size = 10
reps = 30

sims = ['lgssm', 'sv'] # 'toy', 'sv', 'lgssm'
param_dict = {
    # Required arguments:
        'latent_dim': 1,  # latent dimensionality of the data
        'Y': None,  # the observed sequence (i.e. the data)
    # Optional arguments and default values: 
        'inputs': None,  # control inputs (if any)
        'emissions': None,  # the emission model (default: linear transformation plus Gaussian noise)
        'px1_mu': None,  # the Gaussian's prior mean for the initial latent state (default: 0)
        'px1_cov': None,  # the Gaussian's prior covariance for the initial 
                        # latent state (default: identity)
        'kern': None,  # the Gaussian process' kernel (default: Matern 3/2 kernel)
        'Z': None,  # the inducing inputs (default: standard normal samples)
        'n_ind_pts': 50,  # the number of inducing points (ignored if Z is given)
        'mean_fn': None,  # the Gaussian process' mean function (default: the identity function)
        'Q_diag': None,  # the diagonal of the Gaussian process noise's covariance matrix (default: 1)
        'Umu': None,  # the mean of the Gaussian posterior over inducing outputs (default: 0)
        'Ucov_chol': None,  # Cholesky of the covariance matrix of the Gaussian 
                            # posterior over inducing outputs (default: identity - whitening in use)
        'n_samples': 10,  # number of samples from the posterior with which we will compute the ELBO
        'seed': None,  # random seed for the samples
        'jitter': 1e-4,  # amount of jitter to be added to the kernel matrix
        'name': None  # the name of the initialised model in the tensorflow graph
    }

# print(gps.numerics.jitter_level)
methods = ['GP-SSM', 'PR-SSM']

for _ in range(reps):
    
    for sim in sims:

        if sim == 'lgssm':
            obs_noise_covariance = np.eye(1) * 10 
            emissions = GaussianEmissions(
                obs_dim=1,
                R=obs_noise_covariance)
        elif sim == 'toy':
            obs_noise_covariance = np.eye(1)
            multiplier = np.eye(1) * 1./5.
            emissions = GaussianEmissions(
                obs_dim=1, C = multiplier, 
                R=obs_noise_covariance)
        elif sim == 'sv':
            emissions = VolatilityEmissions()

        seed += 1
        obs_train = list()
        for i in range(1):
            if sim == 'lgssm':
                x = 100
                process = LGSSM(true_params=[x], n_obs=10, seed=seed+100*i)
            elif sim == 'toy':
                t1, t2 = 1.0, 0.0
                process = DynamicToyProcess(true_params=[t1,t2], n_obs=10, seed=seed+100*i)
            elif sim == 'sv':
                mu, beta, v0 = 0, 0, 1
                process = StochasticVolatility(true_params=[mu, beta, v0], n_obs=10, seed=seed+100*i)

            # do simulations and state transitions
            for _ in range(steps-1):
                process.step()

        obs_train = np.mean(process.observed, axis=2)

        # preprocess training and testing data
        if sim == 'lgssm':
            latents_train, inf = process.x, 'x'
        elif sim == 'toy':
            latents_train, inf = process.t2, 't2'
        elif sim == 'sv':
            latents_train, inf = process.volatilities, 'volatility'
        # obs_train, obs_test = np.mean(process.observed, axis=2), np.mean(process_test.observed, axis=2)
        # latents_train = np.array(latents_train).reshape((steps, -1))
        obs_train = np.array(obs_train).reshape((steps, -1))
        # latents_test = np.array(latents_test).reshape((steps, -1))
        # obs_test = np.array(obs_test).reshape((steps, -1))

        # generate initial particles from the prior:
        t = {}
        model = process.create_model(observed=process.get_observed()[0])
        t[inf] = model[inf].generate(batch_size=sample_size)
        particles = pd.DataFrame(t)

        emissions.trainable = False
        param_dict['emissions'] = emissions
        param_dict['mean_fn'] = gp.mean_functions.Linear()
        param_dict['Y'] = obs_train

        for meth in methods:
            start_time = time.time()
            if meth == 'GP-SSM':
                model = GPSSM_VCDT(**param_dict)
            elif meth == 'PR-SSM':
                model = PRSSM(**param_dict)

            optimizer = gp.train.AdamOptimizer(0.001) # 0.001
            maxiter = int(3e4)
            # for i in range(100):
            optimizer.minimize(model, maxiter=maxiter)
            print('Value of the variational lower bound for VCDT:', 
                model.compute_log_likelihood())

            session = gp.get_default_session()        
            
            cRMSE = 0
            trajectories = list()
            for i in range(sample_size):
                x0 = np.array(particles[inf][i]).reshape((1, 1)) 
                trajectories.append(session.run(model.sample(50, N=1, x0_samples=x0, return_op=True)).flatten())

            trajectories = np.array(trajectories)
            # print(trajectories)
            trajectories = np.transpose(trajectories)

            mdict, traj = {}, {}
            traj = np.vstack([[np.array([[x[i]] for x in trajectories])] for i in range(sample_size)])
            
            mdict[inf + '_traj'] = traj[:, :, 0].T
            mdict['time'] = (time.time() - start_time) / 60.

            from pathlib import Path
            Path(sim + '/' + meth).mkdir(parents=True, exist_ok=True)
            save_file_name = sim + '/' + meth + '/results-' + str(seed) + '.mat'
            print(save_file_name)
            scipy.io.savemat(save_file_name, mdict=mdict)
