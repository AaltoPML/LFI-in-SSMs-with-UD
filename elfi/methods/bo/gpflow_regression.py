"""This module contains an interface for using the GPy library in ELFI."""

# TODO: make own general GPRegression and kernel classes

import copy
import logging

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.getLogger("GP").setLevel(logging.WARNING)  # GPy library logger


class GPFlowRegression:
    """Gaussian Process regression, using the GPFlow library.
    """

    def __init__(self,
                 parameter_names=None,
                 bounds=None,
                 optimizer="scg",
                 max_opt_iters=50,
                 gp=None,
                 **gp_params):
        """Initialize GPFlowRegression.

        Parameters
        ----------
        parameter_names : list of str, optional
            Names of parameter nodes. If None, sets dimension to 1.
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `{'parameter_name':(lower, upper), ... }`
            If not supplied, defaults to (0, 1) bounds for all dimensions.
        optimizer : string, optional
            Optimizer for the GP hyper parameters
            Alternatives: "scg", "fmin_tnc", "simplex", "lbfgsb", "lbfgs", "sgd"
            See also: paramz.Model.optimize()
        max_opt_iters : int, optional
        gp : GPy.model.GPRegression instance, optional
        **gp_params
            kernel : GPy.Kern
            noise_var : float
            mean_function

        """
        if parameter_names is None:
            input_dim = 1
        elif isinstance(parameter_names, (list, tuple)):
            input_dim = len(parameter_names)
        else:
            raise ValueError("Keyword `parameter_names` must be a list of strings")

        if bounds is None:
            logger.warning('Parameter bounds not specified. Using [0,1] for each parameter.')
            bounds = [(0, 1)] * input_dim
        elif len(bounds) != input_dim:
            raise ValueError(
                'Length of `bounds` ({}) does not match the length of `parameter_names` ({}).'
                .format(len(bounds), input_dim))

        elif isinstance(bounds, dict):
            if len(bounds) == 1:  # might be the case parameter_names=None
                bounds = [bounds[n] for n in bounds.keys()]
            else:
                # turn bounds dict into a list in the same order as parameter_names
                bounds = [bounds[n] for n in parameter_names]
        else:
            raise ValueError("Keyword `bounds` must be a dictionary "
                             "`{'parameter_name': (lower, upper), ... }`")

        self.input_dim = input_dim
        self.bounds = bounds

        self.gp_params = gp_params

        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters

        self._gp = gp
        self.mlls = []

        self._rbf_is_cached = False
        self.is_sampling = False

        self.X = None
        self.Y = None
        self.x_mean = None
        self.x_std = None 
        self.y_mean = None
        self.y_std = None
        custom_config = gpflow.settings.get_settings()
        custom_config.numerics.jitter_level = 1e-8



    def __str__(self):
        """Return GPy's __str__."""
        return self._gp.__str__()

    def __repr__(self):
        """Return GPy's __str__."""
        return self.__str__()

    def predict(self, x, noiseless=False):
        """Return the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        noiseless : bool
            whether to include the noise variance or not to the returned variance

        Returns
        -------
        tuple
            GP (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)
        """
        # Ensure it's 2d for GPy
        x = np.asanyarray(x).reshape((-1, self.input_dim))
        x = (x - self.x_mean) / self.x_std
        mean, var = self._gp.predict_y(x)
        return mean * self.y_std + self.y_mean, var * self.y_std


    def predict_mean(self, x):
        """Return the GP model mean function at x.
        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        Returns
        -------
        np.array
            with shape (x.shape[0], 1)
        """
        return self.predict(x)[0]

    def predict_var(self, x):
        """Return the GP model mean function at x.
        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        Returns
        -------
        np.array
            with shape (x.shape[0], 1)
        """
        return self.predict(x)[1]


    def predictive_gradients(self, x):
        """Return the gradients of the GP model mean and variance at x.
        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        Returns
        -------
        tuple
            GP (grad_mean, grad_var) at x where
                grad_mean : np.array
                    with shape (x.shape[0], input_dim)
                grad_var : np.array
                    with shape (x.shape[0], input_dim)
        """
        # Ensure it's 2d for GPy
        x = x.reshape((-1, self.input_dim))
        x = (x - self.x_mean) / self.x_std

        feed_dict = {self.X_placeholder: x}
        grad_mu, grad_var = self.session.run((self.mean_grad, self.var_grad), 
                                              feed_dict=feed_dict)
        # print(grad_mu[0], grad_var[0])
        return grad_mu[0], grad_var[0]

    def predictive_gradient_mean(self, x):
        """Return the gradient of the GP model mean at x.
        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)
        """
        return self.predictive_gradients(x)[0]

    def _init_gp(self, x, y, optimize):
        if self.x_mean is None or self.x_std is None:
            min_x, max_x = map(list,zip(*self.bounds))
            min_x, max_x = np.array(min_x), np.array(max_x)
            self.x_mean = (max_x + min_x) / 2.0
            self.x_std = np.abs(max_x - self.x_mean)
        
        self.y_mean = np.mean(y, 0)
        self.y_std = np.std(y, 0)
        X = (x - self.x_mean) / self.x_std
        Y = (y - self.y_mean) / self.y_std

        cond = False
        if self._gp is not None:
            self._gp.X = X
            self._gp.Y = Y
            cond = False
            
        if optimize == True:
            if self._gp is not None:
                self._gp.clear()
            self._gp = self.build_model(X, Y, conditioning=cond, apply_name=None)
            self.session = self._gp.enquire_session()



    def build_model(self, X, Y, conditioning=False, apply_name=True,
                    noise_var=None, mean_function=None):
        _, D_in = X.shape 
        kernel = gpflow.kernels.RBF(D_in, lengthscales=float(D_in)**0.5, variance=1., ARD=True)
        model = gpflow.models.GPR(X, Y, kern=kernel)
        model.likelihood.variance = 0.1
        model.compile()

        self.X_placeholder = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
        Fmu, Fvar = model._build_predict(self.X_placeholder)
        self.mean_grad = tf.gradients(Fmu, self.X_placeholder)
        self.var_grad = tf.gradients(Fvar, self.X_placeholder)

        return model


    def update(self, x, y, optimize=False):
        """Update the GP model with new data.
        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize hyperparameters.
        """
        # Must cast these as 2d for GPy
        X = x.reshape((-1, self.input_dim))

        if len(y.shape) == 1:
            Y = y.reshape((-1, 1))
        else:
            Y = y.reshape((-1, y.shape[1]))
        

        if self.X is None or self.Y is None:
            self.X = X
            self.Y = Y
        else:
            self.X = np.r_[self.X, X]
            self.Y = np.r_[self.Y, Y]

        self._init_gp(self.X, self.Y, optimize)

        # print(self.X)
        # print("Optimize? (dgp_regression.py)" + str(optimize))

        if optimize:
            self.optimize()

        if self._gp is not None:
            self.mlls.append(self._gp.compute_log_likelihood())
            print('\nMLL of the iteration: ' + str(self.mlls[-1]) + '\n')
            self._gp.anchor(self.session)


    def optimize(self):
        """Optimize GP hyperparameters."""
        logger.debug("Optimizing GP hyperparameters")
        try:
            gpflow.train.ScipyOptimizer().minimize(self._gp)
        except np.linalg.linalg.LinAlgError:
            logger.warning("Numerical error in GP optimization. Stopping optimization")

    def plot_mlls(self):
        x = list()
        for i in range(0, len(self.mlls)):
            x.append(i+1)
        plt.xticks(np.arange(min(x), max(x)+1))
        plt.grid(color='grey', linestyle='-', linewidth=0.5)
        plt.plot(x, self.mlls, color='blue', label='LogLik')
        plt.legend(loc='upper left')
        return


    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        if self._gp is None:
            return 0
        return self._gp.num_data

    '''@property
    def X(self):
        """Return input evidence."""
        raise ValueError
        return self.X

    @property
    def Y(self):
        """Return output evidence."""
        return self.Y'''

    @property
    def noise(self):
        """Return the noise."""
        return self._gp.Gaussian_noise.variance[0]

    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    def copy(self):
        """Return a copy of current instance."""
        kopy = copy.copy(self)
        if self._gp:
            kopy._gp = self._gp.copy()

        if 'kernel' in self.gp_params:
            kopy.gp_params['kernel'] = self.gp_params['kernel'].copy()

        if 'mean_function' in self.gp_params:
            kopy.gp_params['mean_function'] = self.gp_params['mean_function'].copy()

        return kopy

    def __copy__(self):
        """Return a copy of current instance."""
        return self.copy()
