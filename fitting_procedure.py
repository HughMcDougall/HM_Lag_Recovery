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

    #=Unpack data and params=

    T, Y, E= data['T'], data['Y'], data['E']

    tau = jnp.exp(params['log_tau'])
    sigma_c = jnp.exp(params['log_sigma_c'])

    #If no bands provided, assume continuum only
    if 'bands' in data.keys():
        bands = data['bands']
        cont_only  = False
        Nbands = jnp.max(bands)
    else:
        bands = jnp.zeros_like(T,dtype='int32')
        Nbands = 1
        cont_only = True


    means = params['means']

    if not cont_only:
        line_lags = params['lags']
        line_amps  = params['amps']

    #------------
    #Apply data tform
    #Offset, lag, scale
    Y /= jnp.where(bands == 0, sigma_c, 1) # Scale Continuum
    E /= jnp.where(bands == 0, sigma_c, 1)

    if not cont_only:
        T -= jnp.where(bands>0, line_lags[bands-1] , 0 ) #Apply Lags

        Y /= jnp.where(bands > 0, line_amps[bands - 1], 1) #Scale Line Signal & Errors
        E /= jnp.where(bands > 0, line_amps[bands - 1], 1)

    Y-=means[bands]
    mean = lambda t: 0

    #------------
    #Sort data into gp friendly format
    sort_inds = jnp.argsort(T)

    #Make GP
    kernel = basekernel(scale = tau)
    gp = GaussianProcess(
            kernel,
            T[sort_inds],
            diag=E[sort_inds]**2,
            mean=mean,
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
    #Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c',   numpyro.distributions.Uniform(-2.3,2.3))
    log_tau     = numpyro.sample('log_tau',       numpyro.distributions.Uniform(2,8))

    #Find maximum number of bands in modelling
    Nbands = jnp.max(data['bands'])+1

    #Lag and scaling of respone lines
    lags = numpyro.sample('lags', numpyro.distributions.Uniform(0,  1000),  sample_shape=(Nbands-1,))
    amps = numpyro.sample('amps', numpyro.distributions.Uniform(0,  10),    sample_shape=(Nbands-1,))

    #Means
    means = numpyro.sample('means', numpyro.distributions.Uniform(-100,100), sample_shape=(Nbands,))

    params = {
        'log_tau': log_tau,
        'log_sigma_c': log_sigma_c,
        'lags': lags,
        'amps': amps,
        'means': means,
    }

    #Build TinyGP Process
    gp, sort_inds = build_gp_single(data, params)

    #Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=data['Y'][sort_inds])


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

def fit_single_source(banded_data, params=None):
    '''
    :param banded_data:
        'T': T,
        'Y': Y,
        'E': E,
        'bands': bands,

    :param params:
        "Ncores": 1,
        "Nchain": 300,
        "Nburn": 200,
        "Nsample": 600,
        "step_size": 1E-2

    :return: as dict of outputs
    '''

    # =======================

    #Read input parameters
    if type(params)==type(None):
        params = dict(default_MCMC_params)
    else:
        params = default_MCMC_params | params

    print(params)

    warnings.filterwarnings("ignore", category=FutureWarning)
    numpyro.set_host_device_count( params["Ncores"] )


    # =======================
    # Choose some common sense initial parameters
    # MISSINGNO - update this to be more general

    Nbands = np.max(banded_data["bands"])+1
    init_params = {
        'log_tau': np.log(400),
        'log_sigma_c': 0,
        'amps': np.ones(Nbands-1),
        'means': np.zeros(Nbands),
    }

    # Construct and run MCMC sampler
    sampler = numpyro.infer.MCMC(
        infer.NUTS(nline_model, init_strategy=infer.init_to_value(values=init_params), step_size=params["step_size"]),
        num_chains=params["Nchain"],
        num_warmup=params["Nburn"],
        num_samples=params["Nsample"],
        progress_bar=params["progress_bar"])

    sampler.run(jax.random.PRNGKey(0), banded_data)

    # =======================
    # Return as dictionary
    output = dict(sampler.get_samples())
    #output.pop('means')

    return(output)
