import argparse
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn import preprocessing

from pathlib import Path

from elfi.examples.LGSSM import *
from elfi.examples.stochastic_volatility import *
from elfi.examples.dynamic_toy_model import *


parser = argparse.ArgumentParser()
parser.add_argument('--type')
parser.add_argument('--tasks')
args = parser.parse_args()

methods = ['bnn', 'blr', 'qehvi', 'bolfi', 'SNPE', 'SNLE', 'SNRE', 'GP-SSM', 'PR-SSM']
exps = ['lgssm', 'toy', 'sv', 'umap', 'gaze']
plot = True
plot_type = args.type # inf, traj
add_meth = '-w' + args.tasks if args.tasks else '-w2' # 2 10 # 6
 
inf_methods = ['bnn', 'blr', 'qehvi', 'bolfi', 'SNPE', 'SNLE', 'SNRE']
pred_methods = ['bnn', 'blr']
traj_methods = ['bnn', 'blr', 'GP-SSM', 'PR-SSM']
legend_labels = {'bnn'+add_meth: 'LMC-BNN', 'blr'+add_meth: 'LMC-BLR', 'qehvi'+add_meth: 'LMC-qEHVI', 
                 'bolfi'+add_meth: 'BOLFI', 'bnn-pred'+add_meth: 'LMC-BNN', 'blr-pred'+add_meth: 'LMC-BLR',
                 'SNPE'+add_meth: 'SNPE', 'SNLE'+add_meth: 'SNLE', 'SNRE'+add_meth: 'SNRE', 'GP-SSM'+add_meth: 'GP-SSM', 'PR-SSM'+add_meth: 'PR-SSM'}
data_sizes = ['', '-s5', '-s10']
ds_labels = {'': '2 sims', '-s5': '5 sims', '-s10': '10 sims'}

cwd = os.path.dirname(os.path.realpath(__file__))
bplot = dict()

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


    
matplotlib.rcParams.update({'font.size': 16})


for sim in exps:
    for ds in data_sizes:
        bplot[ds] = dict()
        fig = plt.gcf()
        fig.set_size_inches(8,4)

        # collect the ground truth for UMAP parameterization
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

        steps = 50    
        for meth in methods:

            temp_res, temp_res_pred, temp_res_traj = list(), list(), list()
            evaluate_pred = meth in pred_methods
            evaluate_inf = meth in inf_methods
            evaluate_traj = meth in traj_methods

            # CHANGE THIS LINE IF YOU WANT To ADJUST THE METHODS MOVING WINDOE
            meth += add_meth

            print(cwd + '/' + sim + '/' + meth + ds + '/')
            paths = Path(cwd+'/'+sim + '/' + meth + ds + '/').glob('*.mat')
            
            if sim == 'umap':
                bounds = np.array([[1, 64], [0, 0.99], [2, 200] ])
                n, traj_var = ['d', 'min_dist', 'n'], None
            elif sim == 'gaze':
                bounds = np.array([[30, 60], [0, 0.2], [0, 0.2]])
                n, traj_var = ['eye_latency', 'ocular_noise', 'spatial_noise'], None
            elif sim == 'sv':
                bounds = np.array([[-2, 2], [-5, 5], [0, 3]])
                n, traj_var = ['mu', 'beta', 'volatility'], 'volatility'
            elif sim == 'toy':              
                bounds = np.array([[-30, 30]])
                n, traj_var = ['t1'], 't1'
            elif sim == 'lgssm':
                bounds = [[0.0, 120.0]]
                n, traj_var = ['x'], 'x'
            else:
                continue

            times = []
            for filename in paths:
                # print(filename)
                f = scipy.io.loadmat(filename)
                times.append(f['time'])
                
                rmse = []
                if evaluate_inf and plot_type == 'inf':
                    for s in range(steps):
                        d = dict()
                        for dim in range(len(n)):
                            d[dim] = f[n[dim]][s]

                        pandas_f = pd.DataFrame(data=d)
                        estimate = f['estimates'][0][s]

                        if sim == 'umap':
                            true_estimate = true_estimates[s]
                        else:
                            true_estimate = f['true_params'][0][s]

                        temp_rmse = 0
        
                        for dim in range(len(n)):
                            temp_rmse +=  1./np.abs(bounds[dim][1]-bounds[dim][0]) * np.sqrt( (true_estimate[dim] - estimate[dim])**2)
                        rmse.append(temp_rmse)
                temp_res.append(np.cumsum(rmse))

                rmse = []
                if evaluate_pred and plot_type == 'pred':
                    for s in range(steps-3):
                        d = dict()
                        for dim in range(len(n)):
                            size = len(f[n[dim]+'_pred'][s])
                            d[dim] = f[n[dim]+'_pred'][s] + np.random.normal(loc=1e-6, scale=1e-6, size=size)
                        
                        pandas_f = pd.DataFrame(data=d)
                        estimate = find_kde_peak(pandas_f)

                        if sim == 'umap':
                            true_estimate = true_estimates[s+3]
                        else:
                            true_estimate = f['true_params'][0][s+3]

                        temp_rmse = 0
                        for dim in range(len(n)):
                            temp_rmse +=  1./np.abs(bounds[dim][1]-bounds[dim][0]) * np.sqrt( (true_estimate[dim] - estimate[dim])**2)
                        rmse.append(temp_rmse)
                temp_res_pred.append(np.cumsum(rmse))

                rmse = []
                if evaluate_traj and plot_type == 'traj':
                    # TODO: generate true trajectory for comparison
                    if sim == 'lgssm':
                        x = 100
                        process = LGSSM(true_params=[x], n_obs=1)
                    elif sim == 'toy':
                        t1 = 0.0
                        process = DynamicToyProcess(true_params=[t1], n_obs=1)
                    elif sim == 'sv':
                        mu, beta, v0 = 0, 0, 1
                        process = StochasticVolatility(true_params=[mu, beta, v0], n_obs=1)
                    
                    if traj_var is not None:
                        for _ in range(steps-1):
                            process.step()

                        if sim == 'lgssm': latents = process.x
                        elif sim == 'toy': latents = process.t1
                        elif sim == 'sv':  latents = process.volatilities
                        latents = np.array(latents).reshape((steps, -1))

                        # prepare trajetory
                        d, dim = dict(), n.index(traj_var)
                        traj = f[traj_var + '_traj']
                        temp_rmse = 0

                        for s in range(steps-3):
                            for j in range(traj.shape[1]):
                                temp_rmse +=  1./np.abs(bounds[dim][1]-bounds[dim][0]) * np.sqrt( ( latents[s][0] - traj[s+1][j] )**2)
                            rmse.append(temp_rmse / traj.shape[1])
                    else:
                        temp_rmse = 0
                        for s in range(steps-3):
                            for j in range(traj.shape[1]):
                                
                                if sim == 'umap': true_estimate = true_estimates[s]
                                else: true_estimate = f['true_params'][0][s]
                                for dim in range(len(n)):
                                    traj = f[ n[dim] + '_traj']
                                    temp_rmse +=  1./np.abs(bounds[dim][1]-bounds[dim][0]) * np.sqrt( ( true_estimate[dim] - traj[s+1][j] )**2)
                            rmse.append(temp_rmse / traj.shape[1])
                    
                temp_res_traj.append(np.cumsum(rmse))

            print('Time: ', mean_confidence_interval(times) ) 
                    
            # Plotting
            if evaluate_inf and plot_type == 'inf':
                temp_res = np.array(temp_res).T
                mean = [ mean_confidence_interval(x)[0] for x in temp_res]
                std = [ mean_confidence_interval(x)[1] for x in temp_res]
                low = [ mean_confidence_interval(x)[2] for x in temp_res]
                high = [ mean_confidence_interval(x)[3] for x in temp_res]

            elif evaluate_pred and plot_type == 'pred':
                temp_res = np.array(temp_res_pred).T
                start = [0, 0, 0]
                mean = start + [ mean_confidence_interval(x)[0] for x in temp_res]
                std = start + [ mean_confidence_interval(x)[1] for x in temp_res]
                low = start + [ mean_confidence_interval(x)[2] for x in temp_res]
                high = start + [ mean_confidence_interval(x)[3] for x in temp_res]

            elif evaluate_traj and plot_type == 'traj':
                temp_res = np.array(temp_res_traj).T
                mean = [ mean_confidence_interval(x)[0] for x in temp_res]
                std = [ mean_confidence_interval(x)[1] for x in temp_res]
                low = [ mean_confidence_interval(x)[2] for x in temp_res]
                high = [ mean_confidence_interval(x)[3] for x in temp_res]
            else:
                continue

            if len(temp_res) != 0:
                if plot:
                    plt.fill_between(range(len(low)), low, high, alpha=0.5)
                    plt.plot(mean, label=legend_labels[meth])
                print('cRMSE (' + plot_type + '): ', mean[-1], std[-1], low[-1], high[-1])
                bplot[ds][meth] = temp_res[-1]
            else:
                bplot[ds][meth] = [0]

        if plot:
            plt.xlabel('$\it{t}$')
            plt.ylabel('$\it{cRMSE}$')
        
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

            from pathlib import Path
            Path(cwd + '/plots/').mkdir(parents=True, exist_ok=True)

            plt.savefig(cwd + '/plots/' + sim + ds + '-' + plot_type, dpi=300)
            plt.close()

    if plot:      
        fig = plt.gcf()
        fig.set_size_inches(8,4)


        rows = []
        if plot_type == 'inf':
            plot_methods = inf_methods
        elif plot_type == 'pred':
            plot_methods = pred_methods
        elif plot_type == 'traj':
            plot_methods = traj_methods

        for ds in data_sizes:
            for meth in plot_methods:
                for val in bplot[ds][meth + add_meth]:
                    rows.append( {'RMSE': val, 'Simulation budget': ds_labels[ds] , 'Methods': legend_labels[meth + add_meth]} )
        data = pd.DataFrame(rows)
      
        sns.boxplot(x = data['Simulation budget'],
                y = data['RMSE'],
                hue = data['Methods'],
                palette = 'muted')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        plt.savefig(cwd + '/plots/' + sim + '-test-' + plot_type, dpi=300)
        plt.close()

        print('Results: ', data)
    

