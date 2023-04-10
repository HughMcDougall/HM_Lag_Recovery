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

from modelling import cont_model, nline_model

#============================================
# ----------------------

default_MCMC_params={
    "Ncores": 1,
    "Nchain": 300,
    "Nburn": 200,
    "Nsample": 600,
    "step_size": 1E-3,
    "progress_bar": True,
    "ns-num_live":0,
    "ns-max_samples":0
}

# ----------------------

def fit_single_source(banded_data, MCMC_params=None, seed = 0):
    '''
    #[MISSINGNO] - Documentation
    '''

    # =======================

    # Read input parameters
    if type(MCMC_params)==type(None):
        MCMC_params = dict(default_MCMC_params)
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
    num_live     = MCMC_params["ns-num_live"]
    max_samples  = MCMC_params["ns-max_samples"]


    # =======================
    # Choose some common sense initial parameters
    Nbands = np.max(banded_data["bands"])+1
    if Nbands ==1:
        init_params = {
            'log_tau': np.log(400),
            'log_sigma_c': 0,
            'means': jnp.zeros(Nbands),
        }
        model = cont_model

    else:
        init_params = {
            'log_tau': np.log(400),
            'log_sigma_c': 0,
            'rel_amps': jnp.ones(Nbands - 1),
            'means': jnp.zeros(Nbands),
        }
        model = nline_model

    # =======================
    # Use nested sampling to acquire modes
    from nested_bootstrap import nested_burnin, nested_transform, nested_burnin_tformed

    if num_live==0:
        num_live  = 50*3*(Nbands+1)*(2*Nbands+1)
    if max_samples ==0:
        max_samples = num_live*10

    print("Acquiring modes with nested sampling with %i live points" %num_live)
    nest_burn = nested_burnin(data, nchains= MCMC_params["Nchain"], num_live_points= num_live, max_samples= num_live*10)
    nest, nest_keys = flatten_dict(nest_burn)
    np.savetxt("./nest_results.dat",nest)
    start_positions = nested_transform(nest_burn,Nbands)
    print("Nested Sampling Done. Acquired keys are:")
    for key in start_positions.keys():
        print("\t"+key)
    # --------

    # =======================
    # Construct and run sampler
    sampler_type = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params), step_size=MCMC_params["step_size"])
    sampler = numpyro.infer.MCMC(
        sampler_type,
        num_chains=MCMC_params["Nchain"],
        num_warmup=MCMC_params["Nburn"],
        num_samples=MCMC_params["Nsample"],
        progress_bar=MCMC_params["progress_bar"])

    print("Doing main NUTS run with %i chains, %i burn-in and %i samples with step size starting at %f" %(Nchain, Nburn, Nsample, stepsize))
    sampler.run(jax.random.PRNGKey(seed), banded_data, init_params=start_positions)

    # =======================
    # Return as dictionary
    output = dict(sampler.get_samples())
    return(output)


#=================================================================
if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict

    print("Starting tests for fitting procedure")

    # load some example data
    cont  = array_to_lc(np.loadtxt("./Data/data_fake/360day/cont.dat"))
    line1 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line1.dat"))
    line2 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line2.dat"))

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

    print("unit tests succesfful")
