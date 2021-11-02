"""Example implementation of a linear Gaussian state-space model."""

from functools import partial

import numpy as np
import scipy.stats as ss
import scipy.io

import elfi

def generate_observation(x, observation_scale, observation_noise, n_obs=10, batch_size=1, random_state=None):
    r"""Generate a observation from the single dimensional LGSSM

    Transition:
    x_t = F * x_{t-1} + v_t
    Observation:
    y_t = G * x_t + w_t

    where F and G are constant matrices, v_t and w_t are Gaussian noises.

    Parameters
    ----------
    x : float, array_like
    n_obs : int, optional
    batch_size : int, optional

    Returns
    -------
    y_t : ndarray
        sorted y_t
    """

    random_state = random_state or np.random
    x = np.asanyarray(x).reshape((-1, 1))
    w = random_state.randn(batch_size, n_obs) * observation_noise
    
    return np.sort(x * observation_scale + w)
    

class LGSSM(elfi.DynamicProcess):
    """Linear Gaussian state-space model.

    Attributes
    ----------
    true_params : list
        true_params[0] : x
    transition_scale : float
    transition_noise : float
    observation_scale : float
    observation_noise : float
    """

    def __init__(self, func=generate_observation, transition_scale=None, transition_noise=None, 
        observation_scale=None, observation_noise=None, bounds=None, summarize=2, **kwargs):
        self.name = 'lgssm'
        self.target_name = 'log_d'
        self.transition_scale = transition_scale or 0.95
        self.transition_noise = transition_noise or 1.0
        self.observation_scale = observation_scale or 1.0
        self.observation_noise = observation_noise or 10
        self.bounds = bounds or [[0.0, 120.0]]
        self.func = partial(func, observation_scale=self.observation_scale, observation_noise=self.observation_noise)
        super(LGSSM, self).__init__(func=self.func, parameter_names = ['x'], summarize=2, **kwargs)
        self.x = [self.true_params[0]]


    def create_model(self, observed, lmbd=None, expon_prior=False):
        """Create model with new observed data and prior bounds
        
        Parameters
        ----------
        observed : arraylike
        lmbd: float
            lambda for exponential prior (inverse scale)
        expon_prior : bool, optional
            set to True of exponential prior is preferred

        Returns
        -------
        model : elfi.ElfiModel
        """

        lmbd = lmbd or 0.1

        model = elfi.DynamicElfiModel()

        if expon_prior:
            priors = [elfi.Prior(ss.expon, 0, 1.0/lmbd , model=model, name='x')]
        else:
            priors = [elfi.Prior(ss.uniform, self.bounds[i][0], self.bounds[i][1] - self.bounds[i][0], model=model, name='x') for i in range(self.param_dim)]
        elfi.Simulator(self.sim_fn, *priors, observed=observed, name='Sim')
        if self.summarize:
            elfi.Summary(partial(np.mean, axis=1), model['Sim'], name='Mean')
            elfi.Summary(partial(np.std, axis=1), model['Sim'], name='Std')
            elfi.Distance('euclidean', model['Mean'], model['Std'], name='dist')
            elfi.Operation(np.log, model['dist'], name='log_d')
        else:
            elfi.Distance('euclidean', model['Sim'], name='dist')
            elfi.Operation(np.log, model['dist'], name='log_d')

        return model


    def update_dynamic(self):
        """Update the true value of the dynamic component for the model.
        Parameters
        ----------
        step : float, optional
        node_priors : dict, optional
            Contain names of nodes as keys and (mean, variance) of new normal prior as values.
        """

        self.true_params[0] = self.true_params[0] * self.transition_scale + self.random_state.randn(1)[0] * self.transition_noise
        self.x.append(self.true_params[0])

