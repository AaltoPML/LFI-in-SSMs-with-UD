'''Implementation of the umap task simulator'''

from functools import partial

import numpy as np
import scipy.stats as ss
import scipy.io

import math
import gym
from gym import spaces
from stable_baselines3 import PPO

import elfi
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from sklearn.svm import SVC
import random
import umap

class UMAPTasks(elfi.DynamicProcess):
    """Implementation of the user simulator that performs three data analysis tasks
    for a clustering problem using the UMAP algorithm. Each task is implemented in 
    their separate functions:

    Task 1: visualization, evaluated through the CV object localization performance
    Task 2: clustering enhancement, evaluated through adjusted rand score and mutual 
        information for clustering vs ground truth.
    Task 3: out-of-sample extension, evaluated through the classifier performance
    
    The simulator takes a task and UMAP parameters, it outputs the pefromance metric
    for a given task. The task dynamics is unkown, and the order of tasks depend on
    the human user preferences. The description of the parameters can be found here:
    https://umap-learn.readthedocs.io/en/latest/parameters.html
    

    Attributes
    ----------
    name : str
    target_name : str
    bounds : arraylike
    true_params : list
        true_params[0] : n -- the neighborhood size to use for local metric 
            approximation
        true_params[1] : d -- the dimension of the target reduced space 
            (fixed for this task);
        true_params[2] : min_dist -- how densely the points packed (layout);
    
    cur_task : int
        this parameter governs the UMAP evaluation; 
    _step : int
    observed : int
        observed (ot in this case, expected) best performance for this task
    repeat_task : int
        number of times the current task is repeated

    """

    def __init__(self, bounds=None, **kwargs):
        # Load data: 1797 images, 8x8 pixels, data description can be found through:
        # https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset
        self.data = load_digits()
        features_num = len(self.data.data[0])

        self.bounds = bounds or np.array([[1, features_num], [0, 0.99], [2, 200] ])
        self.target_name = 'log_d' 
        self.observed = [[0]]

        # no true parameters for this task
        self.name = 'umap'
        super(UMAPTasks, self).__init__(func=self.func, true_params = [None, None, None], \
            parameter_names = ['d', 'min_dist', 'n'], **kwargs)
             
        self._step = 0
        self.w = 1. / float(1. + np.exp(-0.1 * (self._step - 25.)))

        
    def func(self, *params, n_obs=1, batch_size=1, random_state=None):
        """
        Stochastic function that performs simulations, given parameter values
        
        Parameters
        ----------
        params : array, 
            a batch of parameters, shape (batch_size, number_of_parameters)
        n_obs : int (optional), 
            number of observed data, equals to 1 in the experiments
        batch_size : int (optional), 
            number of function runs and results to return
        random_state : int (optional), 
            random seed for the stochastic function

        Returns
        -------
        results : array, 
            contains results of simulation (synthetic datasets) for each 
            parameter set, shape (batch_size, result_dimensions = 2)
        """        
        results = list()
        sim_params = np.array( params ).reshape(self.param_dim, -1)
        batches = sim_params.shape[1]

        if batches == 0:
            return [(0, 0)]

        for i in range(0, batches):
            d = sim_params[0, i]
            min_dist = sim_params[1, i]
            n = sim_params[2, i]

            if (n is None) or (d is None) or (min_dist is None):
                results = [(0, 0)]
                continue
            else:
                n, d = int(n), int(d)

            init = 'random' if (n == 2 and d == 1) else 'spectral'
            model = umap.UMAP(n_neighbors=n, min_dist=min_dist, \
                n_components=d, init=init)
            X_train, X_test, y_train, y_test = train_test_split(self.data.data, \
                self.data.target, test_size=0.3)
            embedding = model.fit_transform(X_train)

            min_cluster = int(len(X_train) / 20.) 
            clusterer = hdbscan.HDBSCAN(min_samples=10, \
                min_cluster_size=min_cluster, algorithm='boruvka_kdtree', gen_min_span_tree=True).fit(embedding)
            U = clusterer.relative_validity_ 
            
            svc = SVC().fit(embedding, y_train)
            P = svc.score(model.transform(X_test), y_test)
            
            batch_result = (U, P)
            results.append(batch_result)

        return np.array(results)


    def discrepancy(self, s, obs=None):
        """
        Euclidean distance for each result dimension with coefficients 
        1 and 0.2 (in a separate function to allow additional prints for
        debugging);

        Parameters
        ----------
        s : array, 
            synthetic datasets
        obs : array, 
            observed dataset
        Returns
        -------
        dis : array, 
            discrepancy between the observed dataset and all synthetic datasets
        """ 
        dis = list()

        for entry in s:
            U_clipped = np.clip(entry[0], 0, 1)
            P = entry[1]

            entry_eval = 1 + (self.w - 1) * U_clipped - self.w * P
            dis.append(entry_eval)
            # print('discrepancy (umap.py) -- U:', U, U_clipped, 'P:', P, ' Dis: ', entry_eval)
        return np.array(dis, dtype=np.float32)


    def create_model(self, observed):
        """
        Create model with new observed data and prior bounds.

        Parameters
        ----------
        observed : array

        Returns
        -------
        model : elfi.ElfiModel
        """
        
        model = elfi.DynamicElfiModel()

        priors = [elfi.Prior(ss.uniform, self.bounds[i][0], 
            self.bounds[i][1] - self.bounds[i][0], model=model, name=self.param_names[i]) 
            for i in range(self.param_dim)]

        elfi.Simulator(self.sim_fn, *priors, name='Sim')
        if self.summarize:
            elfi.Summary(partial(np.mean, axis=1), model['Sim'], name='Mean')
            elfi.Summary(partial(np.std, axis=1), model['Sim'], name='Std')
            elfi.Distance('euclidean', model['Mean'], model['Std'], name='dist')
            elfi.Operation(np.log, model['dist'], name=self.target_name)
        else:
            elfi.Distance(self.discrepancy, model['Sim'], name=self.target_name)
            # elfi.Operation(np.log, model['dist'], observed, name=self.target_name)

        return model


    def update_dynamic(self):
        """
        Update the true value of the dynamic component for the model.
        """
        self._step = self._step + 1
        self.w = 1. / float(1. + np.exp(-0.1 * (self._step - 25.))) # (self._step  + np.sin(self._step)) / 50.
        # print('Update dynamics (true_parameters): ', self.w)
