# -*- coding: utf-8 -*-
# flake8: noqa

"""Engine for Likelihood-Free Inference (ELFI) is a statistical software
package for likelihood-free inference (LFI) such as Approximate Bayesian
Computation (ABC).
"""

import elfi.clients.native

import elfi.methods.mcmc
import elfi.model.tools as tools
from elfi.client import get_client, set_client
from elfi.methods.diagnostics import TwoStageSelection
from elfi.methods.model_selection import *
from elfi.methods.parameter_inference import *
from elfi.methods.post_processing import adjust_posterior
from elfi.model.elfi_model import *
from elfi.model.extensions import ScipyLikeDistribution as Distribution
from elfi.store import OutputPool, ArrayPool
from elfi.visualization.visualization import nx_draw as draw
from elfi.visualization.visualization import plot_params_vs_node
from elfi.methods.bo.gpy_regression import GPyRegression
#from elfi.methods.bo.gpflow_regression import GPFlowRegression
from elfi.methods.bo.mogp import LMC
from elfi.methods.dynamic_parameter_inference import *
from elfi.model.dynamic_elfi_model import DynamicElfiModel
from elfi.model.dynamic_process import DynamicProcess

__author__ = 'ELFI authors'
__email__ = 'elfi-support@hiit.fi'

# make sure __version_ is on the last non-empty line (read by setup.py)
__version__ = '0.7.5'
