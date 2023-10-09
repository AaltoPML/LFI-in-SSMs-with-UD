import argparse
import scipy.io
import numpy as np
import elfi
import torch
import time
import os
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn import preprocessing

import statsmodels.api as sm
from statsmodels.graphics import tsaplots

from elfi.examples.LGSSM import LGSSM
from elfi.examples.stochastic_volatility import StochasticVolatility
from elfi.examples.dynamic_toy_model import DynamicToyProcess
from elfi.examples.umap_tasks import UMAPTasks
from elfi.examples.gaze_selection import GazeSelection

from elfi.methods.bo.mogp import MOGPProblem
from elfi.methods.dynamic_parameter_inference import BayesianRegressor

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
    
cwd = os.path.dirname(os.path.realpath(__file__))

def samples_to_numpy(samples, param_names):
    sample_size = len(samples[0][param_names[0]])
    return np.vstack([[np.array([[sample[name][i] for name in param_names] for sample in samples])] for i in range(sample_size)])


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h, m-h, m+h


def stochastic_optimization(fun, bounds, maxiter=1000, polish=True, seed=0):
    result = differential_evolution(
        func=fun, bounds=bounds, maxiter=maxiter, polish=polish, init='latinhypercube', seed=seed)
    return result.x, result.fun


def find_kde_peak(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.to_numpy())
    nparam_density = stats.kde.gaussian_kde(x_scaled.T)
    pars = len(df.columns)

    temp = [0, 1]
    bounds = []
    for i in range(pars):
        bounds.append(temp)
    
    bounds = np.array(bounds)
    func = lambda a : -nparam_density(min_max_scaler.transform(a.reshape(1, -1)))
    x_min, _ = stochastic_optimization(func, bounds)
    return min_max_scaler.inverse_transform(x_min.reshape(1, -1))[0]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim')
    parser.add_argument('--seed')
    args = parser.parse_args()
    start = time.time()
    
    # set experiment parameters
    steps = 50
    num_tasks = 1
    
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
        par_names, traj_var = ['x'], 'x'
        title = r"$\theta_t$"
    elif sim == 'toy':
        t1 = 0.0
        process = DynamicToyProcess(true_params=[t1], n_obs=n_obs, seed=seed)
        par_names, traj_var = ['t1'], 't1'
        title = r"$\theta_h$"
    elif sim == 'sv':
        mu = 0.0
        beta = 0.0
        v0 = 1.0
        process = StochasticVolatility(true_params=[mu, beta, v0], n_obs=n_obs, seed=seed)
        par_names, traj_var = ['mu', 'beta', 'volatility'], 'volatility'
        title = r"$\theta_v$"
        
    elif sim == 'umap':
        process = UMAPTasks(n_obs=n_obs, seed=seed)
        par_names, traj_var = ['d', 'min_dist', 'n'], None
        
        if sim == 'umap':
            umap_steps = 50
            paths = Path(cwd+'/'+sim+'/rejection_abc/').glob('umap-true-*')
            true_estimates = [None] * umap_steps 
            for filename in paths:
                filename = str(filename)
                i = int(filename.split('-')[-1])
                samples = pd.read_csv(filename)
                true_estimate = find_kde_peak(samples)
                true_estimates[i] = true_estimate
            acf_data = true_estimates
            print('ACF_data: ', acf_data)     
    elif sim == 'gaze':
        eye_latency = 37
        ocular_noise = 0.01
        spatial_noise = 0.09
        process = GazeSelection(epochs=1e5, true_params=[eye_latency, ocular_noise, spatial_noise], n_obs=n_obs, seed=seed)
        par_names, traj_var = ['eye_latency'], 'eye_latency'
        acf_data = [[12 * np.log(t+1) + 37] for t in range(50)]
        title = r"$\theta_{l}$"
        
    
    if sim != 'umap' and sim != 'gaze':
        # ===================================
        # RUN BNN tests
        # ===================================
        observed = process.get_observed()[0]
        model = process.create_model(observed=observed)
        
        t = {}
        sampled_trajectories = []
        for param in process.param_names:
            t[param] = model[param].generate(batch_size=10000)
        sampled_trajectories.append(pd.DataFrame(t).head(n=10))
	
        tkwargs={}
        problem = MOGPProblem(process, num_tasks, ref_point=None, bounds=None).to(**tkwargs)
        regressor = BayesianRegressor(problem.input_dim, problem.input_dim, problem.bounds)
	
        # RUN THE PROBLEM TO GET POSTERIOR
        seqABC = elfi.SequentialRejectionABC(process=process)
        # TODO: 0.001 q, 10000 batch size
        seqABC.fit(steps, batch_size=10000, sample_size=10000, quantile=0.001,seed=seed, bar=False, save_sims=False)
        # seqABC.samples # samples_to_numpy(seqABC.samples, process.param_names)    
        samples = seqABC.samples # TODO: get samples from the true posterior	
        acf_data = process.param_path
    
        # print('Samples: ', samples)
        print('ACF_data: ', acf_data)
        print('Minutes spent: ', (time.time() - start)/60.)
    
        # Bayesian neural network: collect data for training the transition model
        X = [pd.DataFrame.from_records(s, columns=par_names).head(100) for s in samples] #[-2:]]
        x_train, y_train = list(), list()
        for i in range(len(X) - 1):
            x_train.append(X[i])
            y_train.append(X[i + 1])
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)

        # train the model with randomly chosen data
        # samples = 10 000 * 1000
        indices = torch.randint(len(x_train), (10000*100,))
        x_train = x_train[indices]
        y_train = y_train[indices]
        regressor.train_bnn(x_train, y_train)
    
        # SAMPLE TRAJECTORIES
        for step in range(steps-1):
            pred = {}
            pred_samples = regressor.sample(sampled_trajectories[-1].to_numpy()).detach().numpy().transpose()
            for i, name in enumerate(process.param_names):
                pred[name] = np.clip( pred_samples[i], process.bounds[i][0], process.bounds[i][1] )
            sampled_trajectories.append(pd.DataFrame(pred))
        # print('Sampled trajectories:', sampled_trajectories)
        print('Minutes spent: ', (time.time() - start)/60.)
    
        # PREPARE TRAJECTORIES FOR EVALUATION
        mdict = {}
        traj = samples_to_numpy(sampled_trajectories, par_names)
        for i, name in enumerate(par_names):
            mdict[name + '_traj'] = traj[:, :, i].T                
        # traj = mdict[traj_var + '_traj']
        # print('Traj: ', mdict)
    
        # EVALUTE THE TRAJECTORIES
        temp_rmse = 0
        rmse = []
        for s in range(steps-1):
            for j in range(10):                    
                true_estimate = acf_data[s]
            
                for dim in range(len(par_names)):
                    traj = mdict[ par_names[dim] + '_traj']
                    temp_rmse +=  1./np.abs(process.bounds[dim][1]-process.bounds[dim][0]) * np.sqrt( ( true_estimate[dim] - traj[s+1][j] )**2)
            rmse.append(temp_rmse / 10.)
        print('RMSE: ', rmse)
        print('cRMSE: ', np.cumsum(rmse))
        print('CI: ', mean_confidence_interval( np.cumsum(rmse) ))
        
        '''
        # EVALUTE THE TRAJECTORIES
        temp_rmse_list = []
        rmse = []
        for s in range(steps-1):
            temp_rmse_list.append( list() )
            for j in range(10):                    
                if sim == 'umap': 
            	    # TODO: true_estimates?? 
            	    true_estimate = true_estimates[s]
                else: 
            	    true_estimate = acf_data[s]
            	
                temp_rmse = 0
                for dim in range(len(par_names)):
                    traj = mdict[ par_names[dim] + '_traj']
                    temp_rmse +=  1./np.abs(process.bounds[dim][1]-process.bounds[dim][0]) * np.sqrt( ( true_estimate[dim] - traj[s+1][j] )**2)
                temp_rmse_list[-1].append(temp_rmse)
        
        print('temp RMSE: ', temp_rmse_list)
        rmse = [ np.cumsum(x)[-1] for x in temp_rmse_list ] 
        print('RMSE: ', rmse)
        print('CI: ', mean_confidence_interval (rmse))
        # print('cRMSE: ', np.cumsum(rmse))'''
    
    # =========
    if sim != 'umap':   
        # print('ACF: ', sm.tsa.acf(acf_data)) #, nlags=steps+1))
        acf_data = np.array(acf_data)[:, par_names.index(traj_var) ]
        
        fig, ax = plt.subplots(1)
       
        # pd.plotting.autocorrelation_plot( data )
        fig = tsaplots.plot_acf(acf_data, lags=steps-1, ax=ax)
        
        # fig = plt.gcf()
        fig.set_size_inches(7,2)
        
        ax.set_title('')
        ax.set_ylabel(title)
        ax.set_xlabel(r'$t$')
    else:
        title = f"UMAP Autocorrelation" 
        par_titles = [r"$\theta_{d}$", r"$\theta_{dist}$", r"$\theta_{n}$"]
        fig, axs = plt.subplots(3)
        fig.suptitle('')
    	
        for par, ax in zip(par_titles, axs):
            acf_data_par = np.array(acf_data)[:, par_titles.index(par)]
            tsaplots.plot_acf(acf_data_par, lags=steps-1, ax=ax)
            ax.set_title("")
            ax.set_ylabel(par)
        
        plt.xlabel(r'$t$')
    fig.tight_layout()
    plt.savefig(cwd + '/plots/' + sim + '-acf', dpi=300)
    plt.close()
    
    print('Minutes spent: ', (time.time() - start)/60.)
    
