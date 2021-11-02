"""This module contains classes for creating ELFI graphs (`ElfiModel`).

The ElfiModel is a directed acyclic graph (DAG), whose nodes represent
parts of the inference task, for example the parameters to be inferred,
the simulator or a summary statistic.

https://en.wikipedia.org/wiki/Directed_acyclic_graph
"""

import inspect
import logging
import os
import pickle
import re
import uuid
from functools import partial

import scipy.spatial
import scipy.stats as ss
import elfi
from elfi.model.elfi_model import *


__all__ = ['DynamicElfiModel']

logger = logging.getLogger(__name__)
_default_model = None

class DynamicElfiModel(ElfiModel):
    """A small extension to ElfiModel to allow replacing a parameter node with a constant node.
    """

    def __init__(self):
        """Initialize the inference model.
        """

        super(DynamicElfiModel, self).__init__()

    def fix(self, node_values):
        """Fixing nodes by replacing them with constant nodes.
        
        Parameters
        ----------
        node_values : dict
            Contains names of nodes as keys and values to be fixed at as values.
        """
        
        for key, value in node_values.items():
            if key in self.parameter_names:
                n = Constant(value, model=self, name=key+'_')
                self[key].become(n)

    def update_prior(self, node_priors):
        """Updating node priors, only Gaussian prior supported.
        
        Parameters
        ----------
        node_priors : dict, required
            contains node names as keys and (mean, std) tuples as values
        """

        for key, values in node_priors.items():
            if key in self.parameter_names:
                n = elfi.Prior(GaussianPrior, values[0], values[1], model=self, name=key+'_')
                self[key].become(n)


class GaussianPrior(elfi.Distribution):
    @classmethod
    def rvs(cls, mean, std, size=1, random_state=None):
        """Get random variates.

        Parameters
        ----------
        mean : float, required
        std : float, required 
        size : int or tuple, optional
        random_state : RandomState, optional

        Returns
        -------
        arraylike

        """

        samples = ss.norm.rvs(loc=mean, scale=std, size=size, random_state=random_state)
        return samples