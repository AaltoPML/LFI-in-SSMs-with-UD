"""This module contains common inference methods."""

__all__ = ['SequentialRejectionABC', 'SequentialNDE', 'SequentialBOLFI', 'LMCInference']

import logging
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import elfi.client


from elfi.methods.utils import (ModelPrior)
from botorch.utils.transforms import unnormalize, normalize

import torch

from gpytorch import settings 
settings.cholesky_jitter(1e-6)
import botorch
import pymc3 as pm 

#from cbfssm.run.run_template import run_cbfssm 


logger = logging.getLogger(__name__)


class SequentialRejectionABC:
    """Sequential Rejection ABC
    Simple interface to allow performing ABC for each time steps independently and record results
    """
    def __init__(self, process):
        self.process = process
        self.model = self.process.get_model()
        self.parameter_names = self.process.param_names
        self.samples = []
        self.estimates = []
        self.pools = []
        self.rejections = []
        self.discrepancies = []
        self.pool_name = 'pool_' + self.process.name


    def fit(self, steps, batch_size=10000, sample_size=10000, quantile=0.001, seed=1, pool_prefix='', save_sims=True, bar=False):
        pool_name = self.pool_name + '_' + str(seed)

        for i in range(steps):
            try:
                if save_sims:
                    pool = elfi.ArrayPool(outputs=self.model.parameter_names + ['Sim'], name=pool_name, prefix=pool_prefix)
                else:
                    pool = elfi.ArrayPool(outputs=self.model.parameter_names, name=pool_name, prefix=pool_prefix)
            except ValueError:
                pool = elfi.ArrayPool.open(pool_name, prefix=pool_prefix)
            
            rej = elfi.Rejection(self.model[self.process.target_name], batch_size=batch_size, seed=seed, pool=pool)
            pool.save()
            self.rejections.append(rej)
            sample = rej.sample(sample_size, quantile=quantile, bar=False)
            self.samples.append(sample.samples)
            self.discrepancies.append(sample.discrepancies)

            if i < steps - 1:
                self.process.step()

            self.model = self.process.get_model()

                
from sbi.inference import infer
from sbi import utils as utils
import torch
from sklearn import preprocessing
import scipy.stats as stats
from scipy.optimize import differential_evolution

class SequentialNDE:
    """Sequential Neural Density Estimators for time-series
    Simple interface to allow performing SNDE from sbi for each time steps independently and record results
    """

    def __init__(self, process, discr=False):
        self.process = process
        self.model = self.process.get_model()
        self.parameter_names = self.process.param_names
        self.samples = []
        self.estimates = []
        self.pools = []
        self.discr = discr

    def simulator(self, parameters):
        result = self.process.func(parameters.numpy()) #self.model[self.process.target_name].generate(with_values=dict_values)
        if self.discr == True:
            result = self.process.discrepancy(result)
        return result
        
    def fit(self, steps, meth, budget=10, num_posterior_samples=1000):
        '''
        Paramaeters
        ----------
        meth : str
            type of the NDE: SNPE, SNLE, SNRE
        '''
        simulator = self.simulator
        prior_min = [self.process.bounds[i][0] for i in range(self.process.param_dim)]
        prior_max = [self.process.bounds[i][1] for i in range(self.process.param_dim)]
        prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))
        posterior = infer(simulator, prior, method=meth, num_simulations=budget)

        for i in range(steps):
            obs = self.process.discrepancy(self.process.observed[-1]) if self.discr else self.process.observed[-1]
            samples = posterior.sample((num_posterior_samples,), x=obs)
            samples = samples.numpy()
            
            dict_samples = dict()
            dict_estimate = dict()
            for i in range(len(samples[0])):
                name = self.process.param_names[i]
                dict_samples[name] = [sample[i] for sample in samples]
            
            dict_estimate = self.find_kde_peak(pd.DataFrame(dict_samples)) #np.mean(dict_samples[name])
                    
            self.samples.append(dict_samples)
            self.estimates.append(dict_estimate)

            if i < steps - 1:
                self.process.step()

            self.model = self.process.get_model()

    def stochastic_optimization(self, fun, bounds, maxiter=1000, polish=True, seed=0):
        result = differential_evolution(
            func=fun, bounds=bounds, maxiter=maxiter, polish=polish, init='latinhypercube', seed=seed)
        return result.x, result.fun

    def find_kde_peak(self, df):
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
        x_min, _ = self.stochastic_optimization(func, bounds)
        return min_max_scaler.inverse_transform(x_min.reshape(1, -1))[0]


from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression

class SequentialBOLFI:
    """Sequential Bayesian Optimization for Likelihood-Free Inference (Sequential BOLFI)
    Simple interface to allow performing BOLFI for each time steps and record results
    """

    def __init__(self, process):
        self.process = process
        self.model = self.process.get_model()
        self.parameter_names = self.process.param_names
        self.samples = []
        self.estimates = []
        self.bolfis = []
        self.pools = []
        self.training_points = {}
        self.simumlations = []
        self.bounds = dict(zip(self.process.param_names, self.process.bounds))

    def fit(self,  
        steps,
        num_initial_evidence=20, 
        num_acquisition_points=5, 
        sample_size=10000, 
        seed=1):
        """Fitting BOLFI for each time step
        Parameters
        ----------
        target_name : str, required
        steps : int, required
        bounds : dict, required
        fix_nodes : dict, optional
            for testing purpose only, nodes specified in fix_nodes should be excluded from bounds
        acq_noise_var : list, optional
        num_initial_evidence : int, optional
        n_evidence : int, optional
        sample_size: int, optional
        seed : int, optional
        """

        acq_noise_var = [0.1] * self.process.param_dim
        bounds = dict(zip(self.process.param_names, tuple(self.process.bounds)))
        for step in range(steps):
            if self.training_points:
                x = self.bolfis[-1].target_model.X
                t = self.training_points.copy()

                y = self.model.generate(batch_size=t['Sim'].shape[0], with_values=t)[self.process.target_name]
    
                if y.ndim == 1:
                    y = np.expand_dims(y, axis=1)
                    
                t[self.process.target_name] = y
                num_initial_evidence = t
                n_evidence = len(y) + num_acquisition_points
            else:
                n_evidence = num_initial_evidence + num_acquisition_points

            pool = elfi.OutputPool(outputs=self.model.parameter_names + ['Sim'])

            target_model = GPyRegression(self.model.parameter_names, bounds=bounds)
            acq =  LCBSC(target_model, prior=ModelPrior(self.model), noise_var=acq_noise_var, 
                            exploration_rate=10, seed=seed)             
            bolfi = elfi.BOLFI(self.model[self.process.target_name], batch_size=num_acquisition_points, initial_evidence=num_initial_evidence,
                           update_interval=num_acquisition_points, target_model=target_model, acquisition_method=acq, seed=seed, pool=pool)

            post = bolfi.fit(n_evidence=n_evidence, bar=False) 
            self.bolfis.append(bolfi)
            self.pools.append(pool)
            
            new_training_points = {}
            for name in self.process.param_names:
                new_training_points[name] = np.hstack([pool.stores[name][i] for i in pool.stores[name].keys()])

            new_training_points['Sim'] = np.vstack([pool.stores['Sim'][i] for i in pool.stores['Sim'].keys()])

            if step == 0:
                for key in new_training_points.keys():
                    self.training_points[key] = new_training_points[key]           
            else:
                for key in new_training_points.keys():
                    self.training_points[key] = np.concatenate((self.training_points[key], new_training_points[key]))

            self.estimates.append(post.estimate)
            self.samples.append(self.importance_sampling(bolfi, self.process.param_names, N=sample_size))
            
            if step < steps - 1:
                self.process.step()
                self.model = self.process.get_model()
    
    def importance_sampling(self, bolfi, parameter_names, N=10000):
        """Importance sampling for posterior
        """
        post = bolfi.extract_posterior()
        theta = post.prior.rvs(size=N)

        if theta.ndim == 1:
            theta = theta.reshape(theta.shape[0], 1)
            
        weights = post._unnormalized_likelihood(theta)
        n_weights = weights / np.sum(weights)

        # importance weighted resampling
        resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
        theta_resampled = theta[resample_index,:]
        theta_df = pd.DataFrame.from_records(theta_resampled, columns=parameter_names)

        return theta_df


from elfi.methods.bo.mogp import LMC, MOGPProblem, LMCPosterior, optimize_qehvi_and_get_observation
from botorch.utils.transforms import unnormalize, normalize

                
class LMCInference:
    """Class for inference with LMC multi-output GP
    
    Attributes
    ==========
    tkwargs : dict
        torch keyword arguments
    num_tasks : int
    num_latents : int
    num_inducing_points : int
    lmc : LMC
    problem : MOGPProblem
    hypervolumes : list
    acq_vals : list
        recorded acquition values
    acq_funcs : list
        copies of acquisition functions
    lmc_state : list
        list of state dict of lmc for each step
    training_points : list
        recorded training points for each step    
    """

    def __init__(self, process, num_tasks, num_latents, num_inducing_points=50, learn_inducing_locations=True, 
                    ref_point=None, bounds=None, tkwargs={}):
        """Constructor for LMC Inference
        
        Parameters
        ==========
        num_tasks : int
            number of tasks or time steps
        num_latents : int
            number of latent GPs
        num_inducing_points : int, optional
            number of inducing points for each latent GP
        learn_inducing_locations : bool, optional
            set to False to keep inducing point location
        ref_point : torch.Tensor or arraylike, optional
            reference point for MOGPProblem
        bounds : torch.Tensor, optional
            bounds for each parameter input, shape 2 x param_dim
        tkwargs : dict, optional
            torch keyword arguments, settings for device and/or dtype
        **kwargs
            keyword arguments for SequentialInference
        """

        self.process = process
        self.model = self.process.get_model()
        self.parameter_names = self.process.param_names
        self.samples = []
        self.estimates = []
        self.pools = []

        self.tkwargs = tkwargs or {'dtype': torch.float, 'device': torch.device('cpu')}
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.num_inducing_points = num_inducing_points
        self.problem = MOGPProblem(self.process, self.num_tasks, ref_point=ref_point, bounds=bounds).to(**self.tkwargs)
        self.lmc = self._lmc_instance(learn_inducing_locations=learn_inducing_locations)
        self.lmc_states = []
        self.training_points = []
        self.optimals = []
        self.losses = []
        self.posteriors = []

        self.predictions = []
        self.acquired_points = []

        self.sampled_trajectories = []
        

    def fit(self, steps, num_initial_evidence=50, num_acquisition_points=5, acq_method='qehvi',
        lmc_epochs=2000, lmc_posterior_samples=10000, prediction_samples=10000, fit_option='full'):
        """Fit the LMC for each step in steps
        
        Parameters
        ==========
        steps : int, required
        num_initial_evidence : int, optional
            number of initial evidence to generate from prior
        num_acquisition_points : int, optional
            number of acquisition points per inference step
        acq_method : string , optionl
            method for getting acuqisition functions: 'qehvi', 'blr', 'bnn'
        lmc_epochs : int, optional
            number of epochs to train with the initial evidence
        lmc_posterior_samples : int, optional
            number of posterior samples
        prediction_samples : int, optional
            number of prediction per inference step
        fit_option : str, optional
            option for fitting, 'full' as default to reset and retrain every parameters, 'linear' to reset and retrain linear LMC coefficient parameters only

        """
        t = {}
        observed = self.process.get_observed()[0]
        model = self.process.create_model(observed=observed)
        # print(self.process.param_names)
        for param in self.process.param_names:
            t[param] = model[param].generate(batch_size=prediction_samples)
        data = pd.DataFrame(t)
        self.sampled_trajectories.append(data.head(n=10))

        if acq_method == 'bnn':
            self.regressor = BayesianRegressor(self.problem.input_dim, self.problem.input_dim, self.lmc.bounds)

        for step in range(steps):
            # Intitial training points
            if self.training_points:
                chosen_rows = np.arange(self.training_points[-1][0].size(0))
                num_initial_evidence = len(chosen_rows)
                training_points = (self.training_points[-1][0][chosen_rows], self.training_points[-1][1][chosen_rows, 1:], self.training_points[-1][2][chosen_rows])
                
                if self.acquired_points:
                    train_x, train_y, train_sim = self.problem.get_evidence(num_initial_evidence, training_points, self.acquired_points[-1])
                    # print('Acquired points: ', self.acquired_points[-1])
                    # print('Current Y: ', train_y)
                else:
                    train_x, train_y, train_sim = self.problem.get_evidence(num_initial_evidence, training_points)
           
            else:
                train_x, train_y, train_sim = self.problem.get_evidence(num_evidence=num_initial_evidence)

            # Train LMC model
            self.lmc.train_lmc(train_x, train_y, num_epochs_max=lmc_epochs, **self.tkwargs)

            # Saving LMC state and training points
            self.lmc_states.append(copy.deepcopy(self.lmc.state_dict()))
            self.training_points.append((train_x, train_y, train_sim))

            min_ind = train_y.argmin(-2)
            min_x = train_x[min_ind]

            # Updating optimals, losses, and posteriors
            self.update_optimal(min_x, step, sample_size=lmc_posterior_samples)

            if step == steps - 1:
                break

            if acq_method == 'qehvi':            
                # number of samples for the sampler for acquisition process
                mc_sample_size = 128
                
                # Acquire and add new training points to current points, retrain the LMC
                new_x = optimize_qehvi_and_get_observation(lmc=self.lmc, problem=self.problem, train_obj=train_y, 
                                                                    mc_sample_size=mc_sample_size, batch_size=num_acquisition_points, **self.tkwargs)
                self.acquired_points.append(new_x)

            elif acq_method == 'blr':
                # Bayesian linear regression for prediction (get t+1 from t, t-1 and t-2)
                if step + self.num_tasks >= 3:
                    X = [s.head(prediction_samples) for s in self.samples[-3:]]
                    traces = []
                    pred = pd.DataFrame(np.zeros((prediction_samples, self.process.param_dim)), columns=self.process.param_names)
                    
                    # train simple BLR
                    for i, name in enumerate(self.process.param_names):
                        with pm.Model() as linear_model:
                            pm.glm.GLM(np.vstack([X[t].to_numpy() for t in range(len(X) -1)]), np.hstack([X[t][name].to_numpy() for t in range(1, len(X))]), family='student')
                            traces.append(pm.sample(1000, chains=2, progressbar=False, tune=1000, target_accept=0.85))

                    # get predictions and trajectories
                    t = {}
                    for i, name in enumerate(self.process.param_names):
                        intercept = np.mean(traces[i]['Intercept'])
                        slopes = np.array([np.mean(traces[i]['x' + str(j)]) for j in range(self.process.param_dim)])
                        pred[name] =  np.clip(intercept + np.matmul(X[-1].to_numpy(), slopes), self.process.bounds[i][0], self.process.bounds[i][1])
                        t[name] = np.clip(intercept + np.matmul(self.sampled_trajectories[-1].to_numpy(), slopes), self.process.bounds[i][0], self.process.bounds[i][1])
                    
                    self.sampled_trajectories.append(pd.DataFrame(t))
                    self.predictions.append(pred)
                    self.acquired_points.append(torch.tensor(pred.to_numpy())[np.random.randint(prediction_samples, size=num_acquisition_points)])

            elif acq_method == 'bnn':
                # Bayesian neural network: collect data for training the transition model
                X = [s.head(prediction_samples) for s in self.samples] #[-2:]]
                x_train, y_train = list(), list()
                for i in range(len(X) - 1):
                    x_train.append(X[i])
                    y_train.append(X[i + 1])
                x_train = np.vstack(x_train)
                y_train = np.vstack(y_train)

                # train the model with randomly chosen data
                # indices = torch.randperm(len(x_train))[:prediction_samples]
                indices = torch.randint(len(x_train), (prediction_samples*1000,))
                x_train = x_train[indices]
                y_train = y_train[indices]
                self.regressor.train_bnn(x_train, y_train)
                    
                # get prediction for the next step
                pred = {}
                pred_samples = self.regressor.sample(X[-1].to_numpy()).detach().numpy().transpose()
                #print('Training: ', x_train, y_train)
                for i, name in enumerate(self.process.param_names):
                    pred[name] = np.clip( pred_samples[i], self.process.bounds[i][0], self.process.bounds[i][1] )
                pred = pd.DataFrame(pred)
                
                self.predictions.append(pred)
                pred = pred.drop_duplicates()

                unique_samples = np.minimum( len(pred), num_acquisition_points)
                self.acquired_points.append(torch.tensor(pred.sample(unique_samples).to_numpy()))
            else:
                raise ValueError("Invalid name of the acquisition method: {}".format(acq_method))

            # Reset for next step
            if fit_option == 'linear':
                self.lmc.modify_parameters(parameter_dict={'variational_strategy.lmc_coefficients': torch.randn(self.num_latents, self.num_tasks)})
                self.lmc.linear_grad_only()
            elif fit_option == 'full':
                self.lmc = self._lmc_instance()

            # Increment the step of the simulator
            if step < steps - 1:
                self.problem.step()

        # sample trajectory
        if acq_method == 'bnn':
            for step in range(steps):
                pred = {}
                pred_samples = self.regressor.sample(self.sampled_trajectories[-1].to_numpy()).detach().numpy().transpose()
                for i, name in enumerate(self.process.param_names):
                    pred[name] = np.clip( pred_samples[i], self.process.bounds[i][0], self.process.bounds[i][1] )
                self.sampled_trajectories.append(pd.DataFrame(pred))


    def _lmc_instance(self, num_latents=None, num_tasks=None, learn_inducing_locations=True):
        """Create a new LMC instance
        """        
        num_latents = num_latents or self.num_latents
        num_tasks = num_tasks or self.num_tasks

        inducing_points = torch.rand(num_latents, self.num_inducing_points, self.process.param_dim)
        lmc = LMC(inducing_points=inducing_points, num_latents=num_latents, num_tasks=num_tasks, bounds=self.problem.bounds,
                    y_bounds=self.problem.y_bounds, learn_inducing_locations=learn_inducing_locations).to(**self.tkwargs)
        return lmc


    def get_lmc_optimal(self, lmc, min_point, observed, iterations_min=100, lr=1e-4, rel_tol=1e-6, restarts=1, thresholds=[]):
        """Extract the minimum mean function and the corresponding point point
        """
        lmc_optimal = []
        lmc_loss = []
        lmc_posterior = []

        for i in range(lmc.num_tasks):
            restart = 0
            restart_bests = []
            thresholds_reached = False

            while restart < restarts and not thresholds_reached:

                restart = restart + 1

                if restart <= 10:
                    x = min_point[i]
                else:
                    t = {}
                    model = self.process.create_model(observed=observed[i])

                    for param in self.process.param_names:
                        t[param] = model[param].generate(batch_size=1)
                    x = torch.stack([torch.tensor(t[param]) for param in self.process.param_names], dim=-1).to(**self.tkwargs) # max_x[i] # torch.stack([torch.tensor(t[param]) for param in self.process.param_names], dim=-1).to(**self.tkwargs)
        
                x = normalize(x, bounds=lmc.bounds).float()
                x = x.unsqueeze(0)
                x.requires_grad = True
                optimizer = torch.optim.Adam([x], lr=lr)
                
                decayRate = 0.98
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
                stopping_criterion = botorch.optim.stopping.ExpMAStoppingCriterion(rel_tol=rel_tol)
                stop = False
                iterations = 0

                while not stop or iterations < iterations_min:
                    optimizer.zero_grad()
                    y = lmc(x)
                    predictions = lmc.likelihood(y)
                    mean = predictions.mean
                    loss = mean[0,i].view(1,1)
                    loss.backward()
                    optimizer.step()
                    stop = stopping_criterion.evaluate(fvals=torch.Tensor([loss]))
                    lr_scheduler.step()
                    iterations = iterations + 1

                loss = unnormalize(loss, bounds=lmc.y_bounds[:,1]).float()        
                restart_bests.append(copy.copy(torch.cat((x.view(-1), torch.Tensor([loss])))))
                if len(thresholds) > 0:
                    thresholds_reached = (loss.item() < -thresholds[i])
            
            restart_bests = torch.stack(restart_bests)
            
            if thresholds_reached:
                best = restart_bests[-1]
            else:
                best_index = torch.argmin(restart_bests[:,-1])
                best = restart_bests[best_index]

            lmc_optimal.append( unnormalize(best[:-1], bounds=lmc.bounds).float().detach().numpy() )
            lmc_loss.append(best[-1])
            thr = lmc.likelihood(lmc(best[:-1].unsqueeze(0))).mean
            post = LMCPosterior(lmc, i, threshold=thr, prior=ModelPrior(self.model))
            lmc_posterior.append(post)

        return lmc_optimal, lmc_loss, lmc_posterior


    def update_optimal(self, min_point, step, iterations_min=100, lr=0.01, rel_tol=1e-9, restarts=1, threshold=0.99, sample_size=10000):
        """Updating optimal values and posteriors
        """
        lmc = self._lmc_instance()
        lmc.load_state_dict(copy.copy(self.lmc_states[-1]))
        observed = [self.process.get_observed()[step + task] for task in range(self.num_tasks)] 
        thresholds = np.quantile(self.training_points[step][1].detach().numpy(), threshold, axis=0, interpolation='higher')
        lmc_optimal, lmc_loss, lmc_posterior = self.get_lmc_optimal(lmc, min_point, observed, thresholds=thresholds)

        for i in range(self.num_tasks):
            if step == 0 or step + i >= len(self.optimals):
                self.optimals.append(lmc_optimal[i])
                self.losses.append(lmc_loss[i])
                self.posteriors.append(lmc_posterior[i])
                self.estimates.append(lmc_optimal[i])
                self.samples.append(lmc_posterior[i].sample_lmc_posterior(cols=self.process.param_names, N=sample_size))
            else:
                if lmc_loss[i] < self.losses[step + i]:
                    self.optimals[step + i] = lmc_optimal[i]
                    self.losses[step + i] = lmc_loss[i]
                    self.posteriors[step + i] = lmc_posterior[i]
                    self.estimates[step + i] = lmc_optimal[i]
                    self.samples[step + i] = lmc_posterior[i].sample_lmc_posterior(cols=self.process.param_names, N=sample_size)
        



from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

import torch.nn as nn
import torch.optim as optim

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, bounds):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 256)
        self.blinear2 = BayesianLinear(256, 256)
        self.blinear3 = BayesianLinear(256, output_dim)
        self.bounds = bounds
        
    def forward(self, x):
        x1 = self.blinear1(x)
        x2 = self.blinear2(x1)
        x3 = self.blinear3(x2)
        # output = torch.clamp(x2, min=0, max=1)
        return x3

    def train_bnn(self, x, y, init=False):
        x, y = torch.tensor(x).float(), torch.tensor(y).float()
        x_train = normalize(x, bounds=self.bounds).float()
        y_train = normalize(y, bounds=self.bounds).float()

        optimizer = optim.SGD(self.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        epochs = 1
        ds_train = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=int(x_train.shape[0]/100.), shuffle=True)

        iteration = 0
        for epoch in range(epochs):
            for i, (datapoints, labels) in enumerate(dataloader_train):
                
                optimizer.zero_grad()
                loss = self.sample_elbo(inputs=datapoints, labels=labels, criterion=criterion, sample_nbr=10, complexity_cost_weight=1./datapoints.shape[0])
                loss.backward()
                optimizer.step()

            iteration += 1
            # if iteration % 1 == 0:
            #     print("\n\n\nLoss (Bayesian Neural Network): {:.4f}".format(loss))

        # if init:
        #     print('End of pre-training')


    def sample(self, x):
        # print(x)
        x = torch.tensor(x).float()
        x = normalize(x, bounds=self.bounds).float()
        y_pred = self(x)
        y_pred = unnormalize(y_pred, bounds=self.bounds).float()
        return y_pred


    def evaluate_regression(self, x, y, samples = 100, std_multiplier = 2):
        # create a confidence interval for predicton
        preds = [self(x) for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()
        return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()
