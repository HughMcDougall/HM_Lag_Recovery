'''
fitting_procedure.py

Contains all the utilities we need for running a single line fit

TODO
-Fill out this todo list

HM 3/4
'''

import warnings

import numpy as np

import jax
import jax.numpy as jnp
#import jaxopt

import numpyro
from numpyro import distributions as dist
from numpyro import infer

from tinygp import GaussianProcess, kernels, transforms
import tinygp

from data_utils import _banded_tform
from copy import deepcopy as copy

#============================================
#Utility Funcs
def mean_func(means, X):
    '''
    Utitlity function to take array of constants and return as gp-friendly functions
    '''
    t, band = X
    return(means[band])


#============================================
#Main Working Funcs
#============================================

#============================================
#TinyGP side
def build_gp_single(data, params, basekernel=tinygp.kernels.Exp):
    '''
    Takes banded LC data and params, returns tinygp gaussian process
    :param data:        Banded lc as dictionary of form {T,Y,E,bands}
    :param params:      Parameters to build the gp from as dictionary
    :param basekernel:  Base gaussian kernel to use. Defaults to exponential
    :return:            Returns tinygp gp object and jnp.array of data sorted by lag-corrected time
    '''

    #Unpack data and params
    T, Y, E= data['T'], data['Y'], data['E']
    tau = params["tau"]

    #------------
    #Data must be sorted for gp
    sort_inds = jnp.argsort(T)

    kernel = basekernel(scale = tau)

    #Make GP
    gp = GaussianProcess(
            kernel,
            T[sort_inds],
            diag=E[sort_inds]**2
        )

    out = (gp, sort_inds)
    return(out)


#============================================
#Numpyro Side

def nline_model(data):
    '''
    Main model, to be fed to a numpyro NUTS object, with banded 'data' as an object
    [MISSINGNO] - general params argument for search ranges
    '''

    #----------------------------------
    # Numpyro Sampling
    
    #Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c',   numpyro.distributions.Uniform(-2.3,2.3))
    log_tau     = numpyro.sample('log_tau',       numpyro.distributions.Uniform(2,8))

    #Find maximum number of bands in modelling
    Nbands = jnp.max(data['bands'])+1

    #Lag and scaling of respone lines
    lags = numpyro.sample('lags', numpyro.distributions.Uniform(0,  1000),  sample_shape=(Nbands-1,))
    amps = numpyro.sample('amps', numpyro.distributions.Uniform(0,  10),    sample_shape=(Nbands-1,))

    #Means
    means = numpyro.sample('means', numpyro.distributions.Uniform(-10,10), sample_shape=(Nbands,))
    #----------------------------------
    #Transform data

    tform_params = {
        'tau': jnp.exp(log_tau),
        'lags': jnp.concatenate([   jnp.array([0]),                     lags]),
        'amps': jnp.concatenate([   jnp.array([jnp.exp(log_sigma_c)]),  amps]),
        'means': means,
    }

    #Scale and shift data / create copy

    #tformed_data = _banded_tform(data, tform_params)

    tformed_data = copy(data)

    bands = tformed_data["bands"]
    Y = tformed_data["Y"]
    E = tformed_data["E"]
    T = tformed_data["T"]

    tformed_data["T"] = T   - tform_params["lags"][bands]
    tformed_data["Y"] = (Y  -   means[bands])                / tform_params["amps"][bands]
    tformed_data["E"] = E                                   / tform_params["amps"][bands]


    #----------------------------------
    #Build and sample GP
    #Build TinyGP Process
    gp, sort_inds = build_gp_single(tformed_data, tform_params)

    #Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=tformed_data['Y'][sort_inds])



#============================================
#Main fitting procedure

default_MCMC_params={
    "Ncores": 1,
    "Nchain": 300,
    "Nburn": 200,
    "Nsample": 600,
    "step_size": 1E-2,
    "progress_bar": True
}

def fit_single_source(banded_data, MCMC_params=None):
    '''
    :param banded_data:
        'T': T,
        'Y': Y,
        'E': E,
        'bands': bands,

    :param MCMC_params:
        "Ncores": 1,
        "Nchain": 300,
        "Nburn": 200,
        "Nsample": 600,
        "step_size": 1E-2

    :return: as dict of outputs
    '''

    # =======================

    #Read input parameters
    if type(MCMC_params)==type(None):
        MCMC_params = dict(default_MCMC_params)
    else:
        MCMC_params = default_MCMC_params | MCMC_params

    print(MCMC_params)

    warnings.filterwarnings("ignore", category=FutureWarning)
    numpyro.set_host_device_count( MCMC_params["Ncores"] )


    # =======================
    # Choose some common sense initial parameters
    # MISSINGNO - update this to be more general

    Nbands = np.max(banded_data["bands"])+1
    init_params = {
        'log_tau': np.log(400),
        'log_sigma_c': 0,
        'amps':  jnp.ones(Nbands-1),
        'means': jnp.zeros(Nbands),
    }

    # Construct and run MCMC sampler
    '''
    sampler = numpyro.infer.MCMC(
        infer.NUTS(nline_model, init_strategy=infer.init_to_value(values=init_params), step_size=MCMC_params["step_size"]),
        num_chains=MCMC_params["Nchain"],
        num_warmup=MCMC_params["Nburn"],
        num_samples=MCMC_params["Nsample"],
        progress_bar=MCMC_params["progress_bar"])
    '''
    sampler = numpyro.infer.MCMC(
        infer.SA(nline_model, init_strategy=infer.init_to_value(values=init_params)),
        num_chains=MCMC_params["Nchain"],
        num_warmup=MCMC_params["Nburn"],
        num_samples=MCMC_params["Nsample"],
        progress_bar=MCMC_params["progress_bar"])

    sampler.run(jax.random.PRNGKey(0), banded_data)

    # =======================
    # Return as dictionary
    output = dict(sampler.get_samples())

    return(output)

if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform
    #load some example data
    cont  = array_to_lc(np.loadtxt("./Data/data_fake/cont.dat"))
    line1 = array_to_lc(np.loadtxt("./Data/data_fake/line1.dat"))
    line2 = array_to_lc(np.loadtxt("./Data/data_fake/line2.dat"))

    #Make into banded format
    lcs_banded = lc_to_banded([cont, line1, line2])
    lcs_banded = data_tform(lcs_banded, normalize_tform(lcs_banded))

    #Fire off a short MCMC run
    MCMC_params={
        "Nchain": 1,
        "Nburn": 10,
        "Nsample": 20,
        }

    out = fit_single_source(banded_data = lcs_banded, MCMC_params=MCMC_params)

    print("unit tests succesfful")
