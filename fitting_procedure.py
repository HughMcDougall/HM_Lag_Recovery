'''
fitting_procedure.py

Contains all the utilities we need for running a single line fit

TODO
-Fill out this todo list

HM 3/4
'''

#============================================
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
    NOT USED HERE
    '''
    t, band = X
    return(means[band])


#============================================
#Main Working Funcs
#============================================


@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        t, band = X
        return t

    def observation_model(self, X):
        '''
        :param X = (t, band)
        '''
        t, band = X
        return self.amplitudes[band] * self.kernel.observation_model(t)
#============================================
#TinyGP side
def build_gp_single(data, params, basekernel=tinygp.kernels.quasisep.Exp):
    '''
    Takes banded LC data and params, returns tinygp gaussian process
    :param data:        Banded lc as dictionary of form {T,Y,E,bands}
    :param params:      Parameters to build the gp from as dictionary
    :param basekernel:  Base gaussian kernel to use. Defaults to exponential
    :return:            Returns tinygp gp object and jnp.array of data sorted by lag-corrected time
    '''

    #Unpack data and params
    T, Y, E= data['T'], data['Y'], data['E']
    bands = data['bands']

    tau  = params["tau"]
    amps = params['amps']

    #------------
    #Data must be sorted for gp

    kernel = basekernel(scale = tau)

    multi_kernel = Multiband(
        amplitudes  =   amps,
        kernel  =   kernel,
    )

    #Make GP
    gp = GaussianProcess(
            multi_kernel,
            (T, bands),
            diag=E**2
        )

    out = gp
    return(out)


#============================================
#Numpyro Side

def cont_model(data):
    '''
    Main model, to be fed to a numpyro NUTS object, with banded 'data' as an object
    [MISSINGNO] - general params argument for search ranges
    '''

    # ----------------------------------
    # Numpyro Sampling

    log_sigma_c     = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(-2.5, 2.5))
    log_tau         = numpyro.sample('log_tau', numpyro.distributions.Uniform(5, 7))

    mean = numpyro.sample('mean', numpyro.distributions.Uniform(-10, 10))

    # ----------------------------------
    # Collect Params for tform
    tformed_data = copy(data)
    tformed_data["Y"] -=mean

    tform_params = {"tau":jnp.exp(log_tau),
                    "amps":jnp.array([jnp.exp(log_sigma_c)]),}

    # ----------------------------------
    # Build and sample GP
    # Build TinyGP Process
    gp = build_gp_single(tformed_data, tform_params)

    # Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=tformed_data['Y'])


def nline_model(data):
    '''
    Main model, to be fed to a numpyro NUTS object, with banded 'data' as an object
    [MISSINGNO] - general params argument for search ranges
    '''

    #----------------------------------
    # Numpyro Sampling
    
    #Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c',   numpyro.distributions.Uniform(-2.5, 2.5))
    log_tau     = numpyro.sample('log_tau',       numpyro.distributions.Uniform(5,7))

    #Find maximum number of bands in modelling
    Nbands = jnp.max(data['bands'])+1

    #Lag and scaling of respone lines

    lags = numpyro.sample('lags', numpyro.distributions.Uniform(0,  500),  sample_shape=(Nbands-1,))
    rel_amps = numpyro.sample('rel_amps', numpyro.distributions.Uniform(0,  2),    sample_shape=(Nbands-1,))

    #Means
    means = numpyro.sample('means', numpyro.distributions.Uniform(-10,10), sample_shape=(Nbands,))
    #----------------------------------
    #Collect Params for tform

    tform_params = {
        'tau': jnp.exp(log_tau),
        'lags': jnp.concatenate([   jnp.array([0]),                     lags]),
        'amps': jnp.concatenate([   jnp.array([1]),  rel_amps]) * jnp.exp(log_sigma_c),
        'means': means,
    }

    #----------------------------------
    #Transform data

    '''
    tformed_data = _banded_tform(data, tform_params)

    '''
    #DEBUG - Try tforming manually
    tformed_data = copy(data)

    bands = tformed_data["bands"]

    tformed_data["T"] -=    tform_params["lags"][bands]
    tformed_data["Y"] -=    means[bands]

    sort_inds = jnp.argsort(tformed_data["T"])

    tformed_data["T"] = tformed_data["T"][sort_inds]
    tformed_data["Y"] = tformed_data["Y"][sort_inds]
    tformed_data["E"] = tformed_data["E"][sort_inds]
    tformed_data["bands"] = tformed_data["bands"][sort_inds]

    #----------------------------------
    #Build and sample GP
    #Build TinyGP Process
    gp = build_gp_single(tformed_data, tform_params)

    #Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=tformed_data["Y"])



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

    # Construct and run MCMC sampler
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
    # DEBUG - Try using SA instead of NUTS
    sampler_type = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params), step_size=MCMC_params["step_size"])
    #sampler_type = infer.SA(model, init_strategy=infer.init_to_value(values=init_params))

    sampler = numpyro.infer.MCMC(
        sampler_type,
        num_chains=MCMC_params["Nchain"],
        num_warmup=MCMC_params["Nburn"],
        num_samples=MCMC_params["Nsample"],
        progress_bar=MCMC_params["progress_bar"])

    sampler.run(jax.random.PRNGKey(0), banded_data)

    # =======================
    # Return as dictionary
    output = dict(sampler.get_samples())

    return(output)


#=================================================================
if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict
    #load some example data

    cont  = array_to_lc(np.loadtxt("./Data/data_fake/360day/cont.dat"))
    line1 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line1.dat"))
    line2 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line2.dat"))

    #Make into banded format
    #lcs_banded = lc_to_banded([cont, line1, line2])
    #lcs_banded = lc_to_banded([cont, line1])
    lcs_banded = lc_to_banded([cont])
    #lcs_banded = data_tform(lcs_banded, normalize_tform(lcs_banded))

    #Fire off a short MCMC run
    MCMC_params={
        "Nchain": 1,
        "Nburn": 10,
        "Nsample": 20,
        }

    out = fit_single_source(banded_data = lcs_banded, MCMC_params=MCMC_params)
    results, keys = flatten_dict(out)

    print("unit tests succesfful")
