"""Classes and methods for Multi-output Gaussian process"""

import copy
import logging
import tqdm

import math
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.lazy import lazify
from gpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, VariationalStrategy
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
import botorch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.utils import add_output_dim
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.posteriors.gpytorch import GPyTorchPosterior




class LMC(ApproximateGP):
    """Class for LMC multi-output GP
    
    Attributes
    ----------
    num_latents : int
        number of latent GPs
    num_tasks : int
        number of tasks or time steps
    _num_outputs : int
        number of tasks, for posterior function
    _input_batch_shape : torch.Size
        required for posterior function
    learn_inducing_locations : bool
        set to False to keep inducing location constant
    variational_strategy : gpytorch.variational.VariationalStrategy
        variational strategy for LMC
    outcome_transform : botorch.models.transforms.outcome.OutcomeTransform
        transform outcome of LMC, not fully implemented
    mean_module : gpytorch.means.Mean
        mean module for LMC
    covar_module : gpytorch.kernels.Kernel
        kernel module for LMC
    """

    def __init__(self, inducing_points, num_latents, num_tasks, bounds, standardize=False, learn_inducing_locations=True):
        """Contructor
        Parameters
        ----------
        inducing_points : torch.Tensor, required
            tensor of inducing points with shape num_latent x num_inducing_points x input_dim
        num_latents : int, required
            number of latent GPs
        num_tasks : int, required
            number of tasks or time steps
        bounds : Tensor, required
            bounds for each input dimension, used for unnormalizing
        standardize : bool
            set to True to standardize outcome, not fully implemented!!!
        learn_inducing_locations : bool
            set to False to keep inducing location constant
        """

        # We have to mark the CholeskyVariationalDistribution as batch so that we learn a variational distribution for each task
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.bounds = bounds
        self._num_outputs = self.num_tasks
        self._input_batch_shape = torch.Size([])
        self.learn_inducing_locations = learn_inducing_locations
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]))

        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy 
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = LMCVariationalStrategy(VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations), 
                                                        num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1)

        ApproximateGP.__init__(self, variational_strategy)

        if standardize:
            self.outcome_transform = Standardize(m=num_tasks)

        # The mean and covariance modules should be marked as batch so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = inducing_points.size(-1), batch_shape=torch.Size([num_latents])),
                                                            batch_shape=torch.Size([num_latents]))

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks, rank=num_tasks)
        self.input_dim = inducing_points.size(-1)
        # self.likelihood._set_noise(0.001)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output dimension in batch            
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


    def predict(self, x, minimize=False):
        x = torch.from_numpy(x).float()
        y = self(x)
        predictions = self.likelihood(y)
        mean_x = -predictions.mean if minimize else predictions.mean
        var_x = predictions.variance
        return mean_x.detach().numpy(), var_x.detach().numpy()


    def posterior(self, X, output_indices=None, **kwargs):
        """Return MultitaskMultivariateNormal posterior for acquisition process.
        """
        X =  unnormalize(X, bounds=self.bounds)
        self.eval()  # make sure model is in eval mode
        with botorch.models.utils.gpt_posterior_settings():
            # insert a dimension for the output dimension
            if self.num_tasks >= 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=torch.Size([])
                )
            #mvn = self.variational_strategy(X, prior=True)
            mvn = self(X.float())
        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def linear_grad_only(self):
        """Disable gradient for all parameters except for LMC linear coefficients.
        """

        for name, param in self.named_parameters():
            if name != 'variational_strategy.lmc_coefficients':
                param.requires_grad = False

    def full_grad(self):
        """Enable gradient for all parameters.
        """

        for name, param in self.named_parameters():
            if name == 'variational_strategy.base_variational_strategy.inducing_points':
                if self.learn_inducing_locations:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    def modify_parameters(self, parameter_dict={}):
        """Replace parameter tensors with those specified in the dictionary.
        """

        state_dict = self.state_dict()
        for name, tensor in parameter_dict.items():
            state_dict[name] = tensor

        self.load_state_dict(state_dict)

    def reset_parameters(self):
        """Reset parameters to default state.
        """

        state_dict = self.state_dict()
        for name, tensor in state_dict.items():
            if name == 'variational_strategy.lmc_coefficients':
                state_dict[name] = torch.randn(state_dict[name].size())
            elif name == 'variational_strategy.base_variational_strategy.inducing_points':
                state_dict[name] = torch.rand(state_dict[name].size())
            elif name == 'variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar':
                state_dict[name] = torch.stack([torch.eye(state_dict[name].size(-1)) for _ in range(state_dict[name].size(0))])
            else:
                state_dict[name] = torch.zeros(state_dict[name].size())

        self.load_state_dict(state_dict)

    def non_likelihood_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'likelihood' not in name:
                params.append(param)
        return params



class Latent(gpytorch.models.ApproximateGP):
    """Small class for individual latent GP.
    """

    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, 
                                                   learn_inducing_locations=True)
        super(Latent, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MOGPProblem(torch.nn.Module):
    """Interface between elfi.DynamicProcess and BoTorch
    
    Attributes
    ----------
    process : elfi.DynamicProcess
    num_task : int
    ref_point : torch.Tensor or arraylike
        reference point to compute Pareto solutions, should be slightly worse than the worse feasible solution
    bounds : torch.Tensor
        bounds for each input dimension of the process, shape 2 x param_dim
    maximize : bool
        set to False for minimization problem
    target_name : str
        target node name of process
    input_dim : int
        dimension of input
    last_step : int
        keeps track of the last step number
    tasks : ndarray
        current step numbers
    """
    
    def __init__(self, process, num_tasks, ref_point=None, bounds=None, maximize=False):
        """Constructor
        
        Parameters
        ----------
        process : elfi.DynamicProcess
        num_tasks : int
        ref_point : torch.Tensor or arraylike, optional
            reference point, take bounds from process by default
        maximizez: bool
            set to True for maximizing problems
        """

        super(MOGPProblem, self).__init__()
        self.maximize = maximize
        if ref_point:
            assert num_tasks == len(ref_point), 'ref_point length must be equal to num_tasks'
        self.ref_point = ref_point or -5 * torch.ones(num_tasks)

        if bounds:
            assert torch.is_tensor(bounds), 'bounds must be torch.Tensor'
            assert bounds.size == torch.Size([2, process.param_dim]), 'bounds must have size 2 x process.param_dim (' + str(process.param_dim) + ')'
            self.bounds = bounds
        else:
            self.bounds = torch.tensor(process.bounds).T

        self.process = process
        self.target_name = self.process.target_name
        self.input_dim = self.process.param_dim
        self.num_tasks = num_tasks
        self.last_step = 0

        for i in range(self.num_tasks-1):
            self.process.step()
            self.last_step = self.last_step + 1
        self.tasks = np.arange(0, num_tasks)

    def _forward(self, x):
        """Return discrepancies given inputs.
        """

        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        t = {}
        num_evidence = x.shape[0]

        for i in range(len(self.process.param_names)):
            t[self.process.param_names[i]] = x[:,i]

        for i in range(self.num_tasks):
            observed = self.process.get_observed()[self.tasks[i]]
            model = self.process.create_model(observed=observed)

            net = model.generate(batch_size=num_evidence, with_values=t)
            new_y = net[self.target_name]
            new_sim = torch.tensor(net['Sim'])

            if i == 0:
                y = torch.tensor(new_y).view(-1,1)
            else:
                y = torch.cat((y, torch.tensor(new_y).view(-1,1)), dim=-1)
        
        if not self.maximize:
            y = -y

        return y, new_sim

    def forward(self, x):
        y, _ = self._forward(x)
        return

    def step(self):
        """Advance 1 step.
        """

        self.process.step()
        self.last_step = self.last_step + 1
        self.tasks = self.tasks + 1

    def update_ref(self, train_y):
        """ Update reference point when new data is available.
        """
        train_y_min = torch.tensor([torch.min(train_y[:,i], dim = 0)[0] for i in range(train_y.size(-1))])
        self.ref_point = train_y_min - 0.1 * torch.abs(train_y_min)

    def get_evidence(self, num_evidence=None, training_points=None, predictions=None):
        """Generate num_evidence evident points from prior.
        """

        train_t = {}
        if training_points:
            new_tasks = self.tasks[1:] if len(self.tasks) > 1 else self.tasks
            train_x = training_points[0]
            train_y = training_points[1]
            train_sim = training_points[2]
            num_evidence = train_x.shape[0]

            for i in range(self.process.param_dim):
                param = self.process.param_names[i]
                train_t[param] = train_x[:,i].detach().numpy()
            train_t['Sim'] = train_sim.detach().numpy()
        else:
            new_tasks = self.tasks

        for i in new_tasks:
            observed = self.process.get_observed()[i]
            model = self.process.create_model(observed=observed)

            if i == 0 or (i == new_tasks[0] and len(new_tasks) != 1):
                for param in self.process.param_names:
                    train_t[param] = model[param].generate(batch_size=num_evidence)

                train_x = torch.stack([torch.tensor(train_t[param]) for param in self.process.param_names], dim=-1)
                net = model.generate(batch_size=num_evidence, with_values=train_t)
                train_y = torch.tensor(net[self.target_name]).view(-1,1)
                train_t['Sim'] = net['Sim']
                train_sim = torch.tensor(net['Sim'])
                if not self.maximize:
                    train_y = -train_y
            else:
                net = model.generate(batch_size=num_evidence, with_values=train_t)
                new_y = torch.tensor(net[self.target_name]).view(-1,1)

                if not self.maximize:
                    new_y = -new_y

                if train_y.shape[1] > 0:
                    train_y = torch.cat((train_y, new_y), dim=-1)
                else:
                    train_y = new_y

        if predictions is not None:
            predicted_t = {}
            num_evidence = predictions.shape[0]

            for i in range(self.process.param_dim):
                param = self.process.param_names[i]
                predicted_t[param] = predictions[:,i].detach().numpy()

            predicted_sim = model.generate(batch_size=num_evidence, with_values=predicted_t)['Sim']
            predicted_t['Sim'] = predicted_sim

            for i in self.tasks:
                observed = self.process.get_observed()[i]
                model = self.process.create_model(observed=observed)

                if i == self.tasks[0]:
                    predicted_y = torch.tensor(model.generate(batch_size=num_evidence, with_values=predicted_t)[self.target_name]).view(-1,1)
                else:
                    new_predicted_y = torch.tensor(model.generate(batch_size=num_evidence, with_values=predicted_t)[self.target_name]).view(-1,1)
                    predicted_y = torch.cat((predicted_y, new_predicted_y), dim=-1)

            if not self.maximize:
                predicted_y = -predicted_y

            train_x = torch.cat((train_x, predictions), dim=0)
            train_y = torch.cat((train_y, predicted_y), dim=0)
            train_sim = torch.cat((train_sim, torch.tensor(predicted_sim)), dim=0)

        return train_x, train_y, train_sim


def train_lmc(lmc, train_x, train_y, num_epochs_max=1000, training_batch_size=None, verbose=False, **tkwargs):
    """Main function for training LMC
    The function uses BoTorch ExpMAStoppingCriterion to stop optimizing if convergence is reached, other wise trains for num_epochs_max epochs.
    
    Parameters
    ----------
    lmc : LMC
    train_x : torch.Tensor
        training input, shape batch_size x input_dim
    train_y : torch.Tensor
        training objectove, shape batch_size x num_tasks
    num_epochs_max: int, optional
        maximum number of training epochs
    training_batch_size : int, optional
    verbose : bool, optional
    tkwargs : dict
        torch keyword arguments
    """

    lmc.train()

    if not torch.is_tensor(train_x):
        train_x = torch.tensor(train_x, **tkwargs)
    if not torch.is_tensor(train_y):
        train_y = torch.tensor(train_y, **tkwargs)
    if hasattr(lmc, 'outcome_transform'):
        train_y, _ = lmc.outcome_transform(train_y)

    training_batch_size = training_batch_size or train_x.size(0)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), shuffle=True, batch_size=training_batch_size)
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, lmc.parameters())}], lr=0.1)
    mll = gpytorch.mlls.VariationalELBO(lmc.likelihood, lmc, num_data=train_y.size(0))
    stopping_criterion = botorch.optim.stopping.ExpMAStoppingCriterion(rel_tol=1e-10)
    stop = False

    if verbose:
        epochs_iter = tqdm.notebook.tqdm(range(num_epochs_max), desc="Epoch", leave=False)
    else:
        epochs_iter = range(num_epochs_max)

    for _ in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = train_loader
        loss_trajectory = torch.Tensor([])
        loss_sum = 0
        for x_batch, y_batch in minibatch_iter:
            x_batch, y_batch = x_batch.to(**tkwargs), y_batch.to(**tkwargs)
            optimizer.zero_grad()

            output = lmc(x_batch.float(), prior=False)
                
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.item()
            if loss_trajectory.size(0) == 0:
                loss_trajectory = torch.Tensor([loss])
            else:
                loss_trajectory = torch.cat((loss_trajectory, torch.Tensor([loss])))
            
        #stop = stopping_criterion.evaluate(fvals=torch.Tensor([loss_sum]))
        if stop:
            if verbose:
                epochs_iter.close()
            break


def optimize_qehvi_and_get_observation(lmc, problem, train_obj, mc_sample_size=128, batch_size=1, noise=None, **tkwargs):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation.
    """
    from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

    if noise:
        q = 1
        if not torch.is_tensor(noise):
            noise = torch.tensor(noise, **tkwargs)
            noise = torch.randn(batch_size, problem.input_dim) * noise
    else:
        q = batch_size    

    # partition non-dominated space into disjoint rectangles
    standard_bounds =torch.ones(2, problem.input_dim).to(**tkwargs)
    standard_bounds[0] = 0
    sampler = SobolQMCNormalSampler(num_samples=mc_sample_size).to(**tkwargs)
    partitioning = NondominatedPartitioning(num_outcomes=problem.num_tasks, Y=train_obj).to(**tkwargs)
    
    acq_func = qExpectedHypervolumeImprovement(
        model=lmc,
        ref_point=problem.ref_point.tolist(),  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    ).to(**tkwargs)
    # optimize
    candidates, acqf_val = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=q,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 10, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    if noise is not None:
        new_x = noise + new_x.repeat(batch_size, new_x.size(1))
        acqf_val = acqf_val.repeat(batch_size, 1)

    new_y, new_sim = problem._forward(new_x)

    return new_x, new_y, new_sim, acqf_val, copy.copy(acq_func)


def get_latents(LMC):
    """Get latent GPs from LMC. Incomplete as likelihood is not available.
    """

    lmc_state_dict = copy.deepcopy(LMC.state_dict())
    latents = []
    for i in range(LMC.num_latents):
        inducing_points = lmc_state_dict['variational_strategy.base_variational_strategy.inducing_points'][i]
        latent = Latent(inducing_points)
        latent_state_dict  = latent.state_dict()
        for name, tensor in latent_state_dict.items():
            if name == 'variational_strategy.inducing_points': 
                lmc_name = 'variational_strategy.base_variational_strategy.inducing_points'
                latent_state_dict[name] = lmc_state_dict[lmc_name][i]
            elif name == 'variational_strategy.variational_params_initialized' or name == 'variational_strategy.updated_strategy':
                pass
            else:
                lmc_name = name.replace('_variational_distribution', 'base_variational_strategy._variational_distribution')
                if name == 'likelihood.noise_covar.raw_noise':
                    #pass
                    latent_state_dict[name] = torch.Tensor([0.0]) #lmc_state_dict[lmc_name][i].view(1)
                else:
                    latent_state_dict[name] = lmc_state_dict[lmc_name][i]

        latent.load_state_dict(latent_state_dict)
        latents.append(latent)

    return latents


def sample_lmc_posterior(post, cols, N=100000):
    theta = post.prior.rvs(size=N)

    if theta.ndim == 1:
        theta = theta.reshape(theta.shape[0], 1)
        
    weights = post._unnormalized_likelihood(theta)
    results = list()

    # importance weighted resampling
    n_weights = weights[post.task] / np.sum(weights[post.task])
    resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
    theta_resampled = theta[resample_index,:]
    theta_df = pd.DataFrame.from_records(theta_resampled, columns=cols)
    return theta_df
