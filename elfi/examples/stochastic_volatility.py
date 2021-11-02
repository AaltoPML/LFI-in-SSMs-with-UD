"""Implementation of a stochastic volatility process."""

from functools import partial

import numpy as np
import scipy.stats as ss
import scipy.io

import elfi

def log_return(mu, beta, volatility, n_obs=10, batch_size=1, random_state=None):
    r"""Generate samples from a stochastic volatility model. 
    
    The full model can be found in Barndorff-Nielsen and Shephard 2002, and Chopin et al., 2013
    Log-return 

        y_t = \mu + \beta * v_t + v_t ^ {0.5} * e_t

    where e_t are white noise ~ N(0, 1).

    Parameters
    ----------
    mu : float, array_like
    beta : float, array_like
    volatility : float, array_like
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    y_t : ndarray
        sorted observations
    """
    random_state = random_state or np.random
    mu = np.asanyarray(mu).reshape((-1, 1))
    beta = np.asanyarray(beta).reshape((-1, 1))
    volatility = np.asanyarray(volatility).reshape((-1, 1))
    e_t = random_state.randn(batch_size, n_obs)
    scale = np.sqrt(np.abs(volatility)) + 10**(-5)
    y_t = mu + beta * volatility + scale * e_t
    #if summarize:
    #    return np.hstack((np.mean(y_t, axis=1), np.std(y_t, axis=1)))
    #else:
    return y_t


class StochasticVolatility(elfi.DynamicProcess):
    """Class for stochastic volatility model.

    Attributes
    ----------
    name : str
    target_name : str
    bounds : arraylike
    true_params : list
        true_params[0] : mu
        true_params[1] : beta
        true_params[2] : volatility
    _step : int
    xi : float
    omega_sqr : float
    lmbd : float
    Z : arraylike
    volatilities : arraylike
    """

    def __init__(self, func=log_return, xi=None, omega_sqr=None, lmbd=None, bounds=None, summarize=2, **kwargs):
        self.name = 'sv'
        self.target_name = 'log_d'
        self.bounds = bounds or np.array([[-2, 2], [-5, 5], [0, 3]])

        def sim_func(parameters):
            if len(parameters) == 1:
                parameters = parameters[0]
            return log_return(parameters[0], parameters[1], parameters[2])

        self.func = sim_func
        super(StochasticVolatility, self).__init__(func=func, parameter_names=['mu', 'beta', 'volatility'], summarize=summarize, **kwargs)

        self._step = 0
        self.xi = xi or 0.5
        self.omega_sqr = omega_sqr or 0.0625
        self.lmbd = lmbd or 0.01
        
        self.Z = [ss.gamma.rvs(self.xi ** 2 / self.omega_sqr, scale = self.omega_sqr / self.xi, random_state=self.random_state)]
        self.volatilities = [self.true_params[2]]
        

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
            elfi.Operation(np.log, model['dist'], name=self.target_name)
        else:
            elfi.Distance('euclidean', model['Sim'], name='dist')
            elfi.Operation(np.log, model['dist'], name=self.target_name)

        return model


    def update_dynamic(self):
        """Update the true value of the dynamic component for the model.
        Parameters
        ----------
        node_priors : dict, optional
            Contain names of nodes as keys and (mean, variance) of new normal prior as values.
        """
        
        self._step = self._step + 1
        k = ss.poisson.rvs(self.lmbd * self.xi**2 / self.omega_sqr, random_state=self.random_state)
        c = []
        e = []

        for j in range(k):
            c.append(ss.uniform.rvs(self._step - 1, 1, random_state=self.random_state))
            e.append(ss.expon.rvs(scale=self.omega_sqr/self.xi, random_state=self.random_state))

        z = np.exp(-self.lmbd) * self.Z[self._step - 1] + np.sum([(np.exp(-self.lmbd * (self._step - c[m]))) * e[m] for m in range(k)])
        self.Z.append(z)
        self.true_params[2] = (self.Z[self._step - 1] - self.Z[self._step] + np.sum(e)) / self.lmbd
        self.volatilities.append(self.true_params[2])
    
    
