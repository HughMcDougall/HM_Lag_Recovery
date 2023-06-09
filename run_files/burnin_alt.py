'''
burnin_alt.py

Alternate lower dimensionality nested

HM 9/4
'''

#============================================
import sys
sys.path.append("..")

import data_utils
import eval_and_opt


import numpy as np

from eval_and_opt import loss
from data_utils import lc_to_banded, flatten_dict, array_to_lc
import jax
import jax.numpy as jnp

import matplotlib.pylab as plt
from copy import deepcopy as copy
import config

import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer

import numpyro
import jax.numpy as jnp
import jax
import tinygp

from copy import deepcopy as copy

from modelling import build_gp
from numpyro.infer.util import transform_fn
from numpyro.contrib.nested_sampling import NestedSampler

from modelling import model_1line, model_2line, model_cont

from config import *



#============================================

def model_2line_lagsonly(data, params):
    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    # Continuum properties
    log_sigma_c = params["log_sigma_c"]
    log_tau     = params["log_tau"]
    means_0      = params["means_0"]

    # Lag and scaling of respone lines
    lags_1        = numpyro.sample('lags_1', numpyro.distributions.Uniform(lag_min, lag_max))
    rel_amps_1    = params["rel_amps_1"]
    means_1        = params["means_1"]


    lags_2        = numpyro.sample('lags_2', numpyro.distributions.Uniform(lag_min, lag_max))
    rel_amps_2    = params["rel_amps_2"]
    means_2        = params["means_2"]

    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.array([1, rel_amps_1, rel_amps_2]) * jnp.exp(log_sigma_c)
    lags = jnp.array([0, lags_1, lags_2])
    means= jnp.array([means_0, means_1  , means_2])

    gp_params  = {"tau:": tau,
                  "lags": lags,
                  "amps": amps,
                  "means:": means}

    # ----------------------------------
    # Apply lags and sort data

    T -= gp_params['lags'][bands]
    inds = jnp.argsort(T)

    T       =   T[inds]
    Y       =   Y[inds]
    E       =   E[inds]
    bands   =   bands[inds]

    # ----------------------------------
    # Build and sample GP
    # Build TinyGP Process
    gp = build_gp(T=T, Y=Y, diag=E*E, bands=bands, tau=tau, amps=amps, means=means)

    # Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=Y)


if __name__=="__main__":
    print("Starting unit tests")

    # load some example data
    rootfol = "../Data/real_data/00-B-2925372393/"

    truelag1=150
    truelag2=150

    cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
    line1 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))
    line2 = array_to_lc(np.loadtxt(rootfol + "CIV.dat"))
    #Make into banded format
    data  = lc_to_banded([cont, line1, line2])
    data = data_utils.data_tform(data, data_utils.normalize_tform(data))

    #==============================
    import eval_and_opt
    import jaxopt

    print("---------------------")


    def lossfunc(params,lags = jnp.array([540,540])):
        params = copy(params)
        params = params |  {"lags":lags}
        out = eval_and_opt._loss_nline(data, params, 3)

        #out-=jnp.product(1-jnp.exp(-params["rel_amps"]*50))

        out = -out

        return(out)

    print("---------------------")

    init_params = data_utils.default_params(Nbands=3)
    init_params.pop('lags')

    start_params["log_tau"]=3
    start_params["log_sigma_c"]=0.53
    start_params["means_0"]=-2.3
    start_params["means_1"]=-0.5
    start_params["rel_amps_1"]=0.78
    start_params["rel_amps_2"]=0.06

    f = lambda params: lossfunc(params,jnp.array([180,180]))
    print(init_params, f(init_params))
    print("Making Optimizer")
    opt = jaxopt.ScipyMinimize(fun=f, tol = 1E-15)
    print("---------------------")
    print("Running Optimizer")
    soln = opt.run(init_params)

    print("---------------------")
    print(soln.params, f(soln.params))

    print("Doing NS")
    print("---------------------")

    ns = NestedSampler(model_2line_lagsonly, constructor_kwargs={"num_live_points": 50*5*(2+1),
                                                                 "max_samples": 50*5*(2+1)*100})
    ns.run(jax.random.PRNGKey(0), data, start_params)

    print("First pass done")
    #first_pass = jnp.array([ns.get_samples(jax.random.PRNGKey(0), 1)["lags_1"],ns.get_samples(jax.random.PRNGKey(0), 1)["lags_2"]])
    #print(first_pass)

    if False:
        f = lambda params: lossfunc(params, first_pass)
        soln = opt.run(soln.params)
        opt = jaxopt.ScipyMinimize(fun=f, tol=1E-15)
        ns.run(jax.random.PRNGKey(0), data, soln.params)

    out = ns.get_samples(jax.random.PRNGKey(0), 50*5*(2+1)*100)

    c = ChainConsumer()
    c.add_chain(out)
    c.plotter.plot(extents=config.plot_extents)
    plt.show()


