'''
modelling.py

Contains all of the numpyro sampling and tinyGP relevant functions

HM 2023
'''

# ============================================
import warnings

import numpy as np

import jax
import jax.numpy as jnp
# import jaxopt

import numpyro
from numpyro import distributions as dist
from numpyro import infer

from tinygp import GaussianProcess, kernels, transforms
import tinygp
from config import *

import data_utils
from copy import deepcopy as copy


# ============================================
# Utility Funcs
def mean_func(means, X):
    '''
    Utitlity function to take array of constants and return as gp-friendly functions
    NOT USED HERE
    '''
    t, band = X
    return (means[band])


# ============================================
# Main Working Funcs
# ============================================


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


# ============================================
# TinyGP side
def build_gp(T, Y, diag, bands, tau, amps, means, basekernel = tinygp.kernels.quasisep.Exp):
    '''
    Takes banded LC data and params, returns tinygp gaussian process
    :param data:        Banded lc as dictionary of form {T,Y,E,bands}
    :param params:      Parameters to build the gp from as dictionary
    :param basekernel:  Base gaussian kernel to use. Defaults to exponential
    :return:            Returns tinygp gp object and jnp.array of data sorted by lag-corrected time
    '''

    #Define kernel
    multi_kernel = Multiband(
                        kernel      =   basekernel(scale=tau),
                        amplitudes  =   amps,
                    )

    meanf = lambda X: mean_func(means,X)

    # Make GP
    gp = GaussianProcess(
                        multi_kernel,
                        (T, bands),
                        diag=diag,
                        mean=meanf
                        )
    return(gp)


# ============================================
# Numpyro Side

def model_cont(data):
    '''
    A jit friendly continuum only numpyro model
    '''

    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(log_tau_min, log_tau_max))
    means_0      = numpyro.sample('means_0', numpyro.distributions.Uniform(mean_min, mean_max))

    # ----------------------------------
    # Collect Params for tform
    tau     = jnp.exp(log_tau)
    amps    = jnp.array([jnp.exp(log_sigma_c)])
    means   = jnp.array([means_0])

    gp_params  = {"tau:": tau,
                  "amps": amps,
                  "means:": means}

    # ----------------------------------
    # Build and sample GP
    # Build TinyGP Process
    gp = build_gp(T=T, Y=Y, diag=E*E, bands=bands, tau=tau, amps=amps, means=means)

    # Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=Y)


def model_1line(data):
    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    # Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(log_tau_min, log_tau_max))
    means_0      = numpyro.sample('means_0', numpyro.distributions.Uniform(mean_min, mean_max))
    # Lag and scaling of respone lines

    lags_1        = numpyro.sample('lags_1', numpyro.distributions.Uniform(lag_min, lag_max))
    rel_amps_1    = numpyro.sample('rel_amps_1', numpyro.distributions.Uniform(rel_amp_min, rel_amp_max))
    means_1        = numpyro.sample('means_1', numpyro.distributions.Uniform(mean_min, mean_max))

    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.array([1, rel_amps_1]) * jnp.exp(log_sigma_c)
    lags = jnp.array([0, lags_1])
    means= jnp.array([means_0, means_1])

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

def model_2line(data):
    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    # Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(log_tau_min, log_tau_max))
    means_0      = numpyro.sample('means_0', numpyro.distributions.Uniform(mean_min, mean_max))

    # Lag and scaling of respone lines
    lags_1        = numpyro.sample('lags_1', numpyro.distributions.Uniform(lag_min, lag_max))
    rel_amps_1    = numpyro.sample('rel_amps_1', numpyro.distributions.Uniform(rel_amp_min, rel_amp_max))
    means_1        = numpyro.sample('means_1', numpyro.distributions.Uniform(mean_min, mean_max))


    lags_2        = numpyro.sample('lags_2', numpyro.distributions.Uniform(lag_min, lag_max))
    rel_amps_2    = numpyro.sample('rel_amps_2', numpyro.distributions.Uniform(rel_amp_min, rel_amp_max))
    means_2        = numpyro.sample('means_2', numpyro.distributions.Uniform(mean_min, mean_max))

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

#-----------------------------------------------------------
def model_nline(data, Nbands=3):
    '''
    General numpyro model for fitting n-sources
    '''

    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']


    # ----------------------------------
    # Numpyro Sampling

    # Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(log_tau_min, log_tau_max))

    # Lag and scaling of respone lines

    lags        = numpyro.sample('lags', numpyro.distributions.Uniform(lag_min, lag_max), sample_shape=(Nbands - 1,))
    rel_amps    = numpyro.sample('rel_amps', numpyro.distributions.Uniform(rel_amp_min, rel_amp_max), sample_shape=(Nbands - 1,))

    # Means
    means       = numpyro.sample('means', numpyro.distributions.Uniform(mean_min, mean_max), sample_shape=(Nbands,))

    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.concatenate([ jnp.array([1]) , rel_amps]) * jnp.exp(log_sigma_c)

    gp_params  = {"tau:": tau,
                  "lags": jnp.concatenate([jnp.array([0]), lags]),
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


#===========================================
# TESTING

if __name__=="__main__":
    from numpyro.infer import MCMC, SA, NUTS
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict

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
        "Nchain": 1,
        "Nburn": 0,
        "Nsample": 5,
        }

    i = 0
    for data in [banded_2line, banded_1line, banded_cont]:
        #T, Y, diag, bands = data["T"], data["Y"], data["E"]*data["E"], data['bands']
        Nbands = np.max(data['bands'])+1
        if Nbands==1:
            model = model_cont
        else:
            model = model_nline
        i+=1


        sampler = MCMC(SA(model),
                       num_chains=MCMC_params['Nchain'],
                       num_warmup=MCMC_params['Nburn'],
                       num_samples=MCMC_params['Nsample'],
                       )
        sampler.run(jax.random.PRNGKey(0), data)

    print("unit tests succesfful")


