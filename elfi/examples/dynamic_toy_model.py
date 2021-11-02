"""Example implementation of a simple toy dynamic process."""

from functools import partial

import numpy as np
import scipy.stats as ss
import scipy.io

import elfi

def generate_observation(t1, scale=1, n_obs=10, batch_size=1, random_state=None):
    r"""Generate a sequence of samples from a simplied toy model. 
    
    The original model can be found in Gordon et al., 1993, Kitagawa, 1996, and Andrieu et al., 2010
    
        Transition:
        \theta_2^(t) = \theta_2^{(t - 1)} / 2 + 25 \theta_2^{(t - 1)} / [(\theta_2^{(t - 1)})^2 + 1] + 8 cos(1.2t) + v_t
        
        Observation:
            Original:
            y_t = (\theta_2^(t))^2 * scale + w_t * \theta_1
            Our simplified version:
            y_t = \theta_2^(t) * scale + w_t * \theta_1

    where v and w are white noise ~ N(0, 10) N(0, 1).

    Parameters
    ----------
    t1 : float, arraylike
        \theta_1
    x_init : float, optional
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    y_t
        sorted observations
    """

    random_state = random_state or np.random
    t1 = np.asanyarray(t1).reshape((-1, 1))
    w = random_state.randn(batch_size, n_obs) * 10
    
    return (t1 ** 2) * scale + w
    

class DynamicToyProcess(elfi.DynamicProcess):
    """A simple dynamic toy process.

    Attributes
    ----------
    true_params : list
        true_params[0]: t1 (dynamic)
    scale : float
    t1 : list
        trajectory of t1
    _step : int
        current step
    """
    
    def __init__(self, func=generate_observation, scale=None, bounds=None, summarize=2, **kwargs):
        self.name = 'toy_model'
        self.target_name = 'log_d'
        self.scale = scale or 1. / 20.
        self.bounds = bounds = bounds or [[-30, 30]]

        def sim_func(parameters):
            return generate_observation(parameters[0])
        
        self.func = sim_func
        super(DynamicToyProcess, self).__init__(func=func, parameter_names = ['t1'], summarize=summarize, **kwargs)
        self.t1 = [self.true_params[0]]
        self._step = 0


    def create_model(self, observed):
        """Create model with new observed data and prior bounds
        
        Parameters
        ----------
        observed : array

        Returns
        -------
        model : elfi.ElfiModel
        """

        model = elfi.DynamicElfiModel()

        priors = [elfi.Prior(ss.uniform, self.bounds[i][0], self.bounds[i][1] - self.bounds[i][0], model=model, name=self.param_names[i]) for i in range(self.param_dim)]
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
        node_priors : dict, optional
            Contain names of nodes as keys and (mean, variance) of new normal prior as values.
        """
        
        self._step = self._step + 1
        self.true_params[0] = self.true_params[0] / 2 + 25 * self.true_params[0]/(self.true_params[0] ** 2 + 1) + 8 * np.cos(1.2 * self._step) + self.random_state.randn(1)[0] * np.sqrt(10)
        self.t1.append(self.true_params[0])


