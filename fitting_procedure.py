'''
fitting_procedure.py

Contains functions that are actually called by the user.

HM 3/4
'''

#============================================
import warnings

import numpy as np

import jax
import jax.numpy as jnp

import numpyro
from numpyro import infer
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer.util import transform_fn

from data_utils import flatten_dict

from modelling import model_cont, model_nline, model_1line, model_2line

from config import *
from nested_bootstrap import nested_burnin, nested_transform

#============================================

# ----------------------

def fit_single_source(banded_data, MCMC_params=None, seed = 0, return_nested = False):
    '''
    #[MISSINGNO] - Documentation
    '''

    # =======================

    # Read input parameters
    if type(MCMC_params)==type(None):
        MCMC_params = default_MCMC_params
    else:
        MCMC_params = default_MCMC_params | MCMC_params

    print(MCMC_params)

    warnings.filterwarnings("ignore", category=FutureWarning)
    numpyro.set_host_device_count( MCMC_params["Ncores"] )

    Nchain       = MCMC_params["Nchain"]
    Nburn        = MCMC_params["Nburn"]
    Nsample      = MCMC_params["Nsample"]
    stepsize     = MCMC_params["step_size"]
    progress_bar = MCMC_params["progress_bar"]
    num_live     = MCMC_params["ns_num_live"]
    max_samples  = MCMC_params["ns_max_samples"]


    # =======================
    # Identify the model type / band number

    Nbands = np.max(banded_data["bands"])+1
    if Nbands ==1:
        model = model_cont
    elif Nbands ==2:
        model = model_1line
    elif Nbands ==3:
        model = model_2line
    else:
        raise TypeError("Cannot have more than 2 lines in current fit_procedure()")

    # =======================
    # Use nested sampling for initial pass

    print("Acquiring modes with nested sampling with %i live points" %num_live)
    nest_results    = nested_burnin(data, nchains= Nchain, num_live_points=num_live, max_samples=max_samples)
    start_positions = nested_transform(nest_results) # Transformation required for passing to sampler.run()

    # =======================
    # Use NUTS HMC to refine contours
    sampler_type = infer.NUTS(model, step_size=stepsize)
    sampler = numpyro.infer.MCMC(
            sampler_type,
            num_chains  =   Nchain,
            num_warmup  =   Nburn,
            num_samples =   Nsample,
            progress_bar=   progress_bar
        )

    print("Doing main NUTS run with %i chains, %i burn-in and %i samples with step size starting at %f" %(Nchain, Nburn, Nsample, stepsize))
    sampler.run(jax.random.PRNGKey(seed), banded_data, init_params=start_positions)

    # =======================
    # Return results

    NUTS_output = dict(sampler.get_samples())
    if not return_nested:
        return(NUTS_output)
    else:
        NEST_output = dict(nest_results)
        return(NUTS_output, NEST_output)


#=================================================================
if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict

    print("Starting tests for fitting_procedure")

    # load some example data
    rootfol = "./Data/data_fake/clearsignal/"
    cont  = array_to_lc(np.loadtxt(rootfol+"cont.dat" ))
    line1 = array_to_lc(np.loadtxt(rootfol+"line1.dat"))
    line2 = array_to_lc(np.loadtxt(rootfol+"line2.dat"))

    #Make into banded format
    banded_2line  = lc_to_banded([cont, line1, line2])
    banded_1line  = lc_to_banded([cont, line1])
    banded_cont   = lc_to_banded([cont])

    #Fire off a short MCMC run
    MCMC_params={
        "Nchain": 5,
        "Nburn": 10,
        "Nsample": 20,
        }

    for data in [banded_2line, banded_1line, banded_cont]:
        print("Unit tests for %i lines: " %max(data["bands"]) )
        out = fit_single_source(banded_data = data, MCMC_params=MCMC_params)
        results, keys = flatten_dict(out)

    print("Unit tests done")
