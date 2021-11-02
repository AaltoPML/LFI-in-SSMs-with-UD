"""Simple class for dynamic process"""

from functools import partial

import numpy as np
import scipy.stats as ss
import scipy.io
import copy

import elfi

class DynamicProcess:
    """A simple class for a dynamic process.

    Attributes
    ----------
    true_params : list
        parameters with which the observed data is generated
    parameter_name : list
        list of parameter name. A default should be set for each custom example class
    param_dim : int
        number of input dimension
    n_obs : int
        number of observations of the sequence
    random_state : np.random.RandomState
        random state for the data generation
    param_path : list
        store true parameters for each step
    sim_fn : func
        simulator function
    sumarize : int
        number of summary statistics being used, 0 for none
    model : elfi.DynamicElfiModel
        model of the current step
    observed : list
        list of all observations
    node_prior : dict
        prior specification for node, empty if unused 
    obs_dim : int
        observation dimension
    """

    def __init__(self, func, true_params, parameter_names, n_obs=20, summarize=0, seed=1):
        """Construct the dynamic process object
        Paramenters
        ----------
        func : function, required
            generating function for the dynamic process
        true_params : list, required
            parameters with which the observed data is generated
        parameter_names: list, required
            names of parameters
        n_obs : int, optional
            number of observations of the sequence
        summarize: int, optional
            number of summary statistics, 0 if unused 
        seed_obs : int, optional
            seed fof random state
        """
        
        assert len(true_params) ==  len(parameter_names), 'There must be the same number of true parameters and names'

        self.name = '' or self.name
        self.true_params = true_params
        self.param_names = parameter_names
        self.param_dim = len(self.param_names)
        self.n_obs = n_obs
        self.random_state = np.random.RandomState(seed)
        self.param_path = [copy.copy(self.true_params)]
        self.sim_fn = partial(func, n_obs=self.n_obs, random_state=self.random_state)

        y = self.sim_fn(*self.true_params)
        self.summarize = summarize
        self.model = self.create_model(observed=y)        
        
        self.observed = [y]
        self.node_priors = {}
        if self.summarize:
            self.obs_dim = self.summarize
        else:
            self.obs_dim = self.n_obs


    def get_model(self):
        return self.model

    
    def get_observed(self):
        return np.array(self.observed)
        

    def create_model(self, observed):
        """Create model with new observed data and prior bounds
        Parameters
        ----------
        observed : array

        Returns
        -------
        model : elfi.ElfiModel
        """

        raise NotImplementedError


    def update_dynamic(self):
        """Update dynamic parameters of the process
        """

        raise NotImplementedError


    def step(self, node_priors={}):
        """Advance 1 step.
        """

        self.update_dynamic()
        self.param_path.append(copy.copy(self.true_params))
        y = self.sim_fn(*self.true_params)
        self.model = self.create_model(y)
        self.model.update_prior(node_priors)
        self.observed.append(y)


    def update_prior(self, node_priors={}):
        """Update priors of the model.
        Parameters
        ----------
        node_priors : dict, optional
            Contain names of nodes as keys and (mean, variance) of new normal prior as values.
        """

        self.node_priors = node_priors or self.node_priors
        self.model.update_prior(self.node_priors)


    def save_mat(self, theta, file, start, end, reverse=False):
        """Save process data and observation to Matlab .mat file
        Parameters
        ----------
        data: ndarray
            Contains the data. Shape (time_steps, data_dimension)
        file: str
            Save file location

        Note: For now, only take 1 observation and batch_size of 1 per time steps
        """

        obs = self.get_observed()

        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=1)
        if obs.ndim == 3:
            obs = obs.squeeze(axis=1)

        obs = obs[start:end]

        if self.summarize == 1:
            obs = np.mean(obs, axis=1).reshape(-1,1)
        if self.summarize == 2:
            obs = np.hstack((np.mean(obs, axis=1).reshape(-1,1), np.std(obs, axis=1).reshape(-1,1)))

        if not reverse:
            scipy.io.savemat(file, mdict={'u': theta, 'y': obs})
        else:
            scipy.io.savemat(file, mdict={'u': obs, 'y': theta})

