'''
nested_bootstrap.py
'''

#==================================================

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


#==================================================

def nested_burnin_tformed(data, nchains, num_live_points = 0, max_samples = 0, seed = 0):
    Nbands = jnp.max(data['bands']) + 1
    samples = nested_burnin(data, nchains, num_live_points, max_samples, seed)
    tformed_samples = nested_transform(samples,Nbands)
    return(tformed_samples)


def nested_burnin(data, nchains, num_live_points = 0, max_samples = 0, seed = 0):
    '''
    Performs nested sampling on the data-set to get original contours. These results must be fed through
    nested_transform to be useful to use in a sampler.run(init_params = []) call.
    '''
    Nbands = jnp.max(data['bands']) + 1

    num_modes = ((Nbands-1)*2)**2 + 1
    num_dims  = Nbands*3

    if num_live_points ==0:
        num_live_points = 4*num_modes * (num_dims+1)
    if max_samples == 0:
        max_samples = num_live_points * 4

    #----

    if Nbands == 1:
        model = _nested_model_cont
    elif Nbands ==2:
        model = _nested_model_1line
    elif Nbands ==3:
        model = _nested_model_2line
    else:
        assert False, "nested burn in only current implemented for up to two lines"

    ns = NestedSampler(model, constructor_kwargs={"num_live_points": num_live_points,
                                                  "max_samples": max_samples})
    ns.run(jax.random.PRNGKey(seed), data)

    samples = ns.get_samples(jax.random.PRNGKey(seed), nchains)

    return(samples)

def nested_transform(samples, Nbands):
    '''
    Transforms the samples from nested_burnin to the constrained domain. Returns as dictionary
    '''
    #----
    if Nbands == 1:
        transforms = {
            'log_sigma_c': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-2.5, 2.5).support),
            'log_tau': numpyro.distributions.biject_to(numpyro.distributions.Uniform(5, 7).support),
            'means_0': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-10, 10).support),
        }

    elif Nbands ==2:
        transforms = {
            'log_sigma_c': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-2.5, 2.5).support),
            'log_tau': numpyro.distributions.biject_to(numpyro.distributions.Uniform(5, 7).support),
            'means_0': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-10, 10).support),

            'lags_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(0, 500).support),
            'rel_amps_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(0, 10).support),
            'means_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-10, 10).support),
        }
    elif Nbands ==3:
        transforms = {
            'log_sigma_c': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-2.5, 2.5).support),
            'log_tau': numpyro.distributions.biject_to(numpyro.distributions.Uniform(5, 7).support),
            'means_0': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-10, 10).support),

            'lags_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(0, 500).support),
            'rel_amps_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(0, 10).support),
            'means_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-10, 10).support),

            'lags_2': numpyro.distributions.biject_to(numpyro.distributions.Uniform(0, 500).support),
            'rel_amps_2': numpyro.distributions.biject_to(numpyro.distributions.Uniform(0, 10).support),
            'means_2': numpyro.distributions.biject_to(numpyro.distributions.Uniform(-10, 10).support),
        }
    else:
        assert False, "nested burn in only current implemented for up to two lines"

    #--------
    #Transform samples to constrained domain
    start_positions = transform_fn(transforms, samples, invert=True)
    # --------
    # Assemble starting positions into correct format
    if Nbands == 1:
        out = samples

    elif Nbands ==2:
        out = {"log_tau": samples["log_tau"],
               "log_sigma_c": samples["log_sigma_c"],
               "lags": jnp.array([samples["lags_1"]]).T,
               "rel_amps": jnp.array([samples["rel_amps_1"]]).T,
               "means": jnp.array([samples["means_0"],samples["means_1"]]).T
               }
    elif Nbands ==3:
        out = {"log_tau": samples["log_tau"],
               "log_sigma_c": samples["log_sigma_c"],
               "lags": jnp.array([samples["lags_1"],samples["lags_2"]]).T,
               "rel_amps": jnp.array([samples["rel_amps_1"],samples["rel_amps_2"]]).T,
               "means": jnp.array([samples["means_0"],samples["means_1"], samples["means_2"]]).T
               }

    return(out)


#==================================================


def _nested_model_cont(data):
    '''
    A jit friendly continuum only numpyro model
    '''

    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(-2.5, 2.5))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(5, 7))
    mean        = numpyro.sample('mean', numpyro.distributions.Uniform(-10, 10))

    # ----------------------------------
    # Collect Params for tform
    tau     = jnp.exp(log_tau)
    amps    = jnp.array([jnp.exp(log_sigma_c)])
    means   = jnp.array([jnp.exp(log_sigma_c)])

    gp_params  = {"tau:": tau,
                  "amps": amps,
                  "means:": means}

    # ----------------------------------
    # Build and sample GP
    # Build TinyGP Process
    gp = build_gp(T=T, Y=Y, diag=E*E, bands=bands, tau=tau, amps=amps, means=means)

    # Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=Y)


def _nested_model_1line(data):
    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    # Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(-2.5, 2.5))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(5, 7))
    mean_c        = numpyro.sample('means_0', numpyro.distributions.Uniform(-10, 10))
    # Lag and scaling of respone lines

    lags_1        = numpyro.sample('lags_1', numpyro.distributions.Uniform(0, 500))
    rel_amps_1    = numpyro.sample('rel_amps_1', numpyro.distributions.Uniform(0, 10))
    mean_1        = numpyro.sample('means_1', numpyro.distributions.Uniform(-10, 10))

    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.array([1, rel_amps_1]) * jnp.exp(log_sigma_c)
    lags = jnp.array([0, lags_1])
    means= jnp.array([mean_c, mean_1])

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

def _nested_model_2line(data):
    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    # ----------------------------------
    # Numpyro Sampling

    # Continuum properties
    log_sigma_c = numpyro.sample('log_sigma_c', numpyro.distributions.Uniform(-2.5, 2.5))
    log_tau     = numpyro.sample('log_tau', numpyro.distributions.Uniform(5, 7))
    mean_c        = numpyro.sample('means_0', numpyro.distributions.Uniform(-10, 10))

    # Lag and scaling of respone lines

    lags_1        = numpyro.sample('lags_1', numpyro.distributions.Uniform(0, 500))
    rel_amps_1    = numpyro.sample('rel_amps_1', numpyro.distributions.Uniform(0, 10))
    mean_1        = numpyro.sample('means_1', numpyro.distributions.Uniform(-10, 10))

    lags_2        = numpyro.sample('lags_2', numpyro.distributions.Uniform(0, 500))
    rel_amps_2    = numpyro.sample('rel_amps_2', numpyro.distributions.Uniform(0, 10))
    mean_2        = numpyro.sample('means_2', numpyro.distributions.Uniform(-10, 10))

    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.array([1, rel_amps_1, rel_amps_2]) * jnp.exp(log_sigma_c)
    lags = jnp.array([0, lags_1, lags_2])
    means= jnp.array([mean_c, mean_1  , mean_2])

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

    for data,model in zip([banded_cont, banded_1line, banded_2line],
                          [_nested_model_cont, _nested_model_1line, _nested_model_2line]):
        print("Unit tests for %i lines: " %max(data["bands"]) )
        nest_results = nested_burnin(data, 300, num_live_points=1000, max_samples=2000)
        nest_results_tformed = nested_transform(nest_results, np.max(data['bands'])+1)

    from chainconsumer import ChainConsumer
    import matplotlib.pylab as plt
    c= ChainConsumer()
    print(nest_results)
    c.add_chain(nest_results)
    c.plotter.plot()
    plt.show()


    print("unit tests succesfful")