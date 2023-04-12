'''
nested_burnin.py
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

from modelling import model_1line, model_2line, model_cont

from config import *


#==================================================

def nested_burnin_tformed(data, nchains, num_live_points = 0, max_samples = 0, seed = 0):
    Nbands = jnp.max(data['bands']) + 1
    samples = nested_burnin(data, nchains, num_live_points, max_samples, seed)
    tformed_samples = nested_transform(samples)
    return(tformed_samples)


def nested_burnin(data, nchains, num_live_points = 0, max_samples = 0, seed = 0):
    '''
    Performs nested sampling on the data-set to get original contours. These results must be fed through
    nested_transform to be useful to use in a sampler.run(init_params = []) call.
    '''
    Nbands = jnp.max(data['bands']) + 1

    try:
        nchains_int = int(nchains)
    except:
        try:
            nchains_int = [int(num_samples) for num_samples in nchains]
        except:
            raise TypeError("Bad type input to nested_burnin for nchains")

    num_modes = ((Nbands-1)*2)*((Nbands-1)*2) + 1
    num_dims  = 3*Nbands
    if num_live_points ==0:
        num_live_points = 50*num_modes * (num_dims+1)
    if max_samples == 0:
        max_samples = num_live_points * 20
    print("In nested_burnin:\t num_live:\t%i\tmax_samples:\t%i" % (num_live_points, max_samples))

    if type(nchains_int)==int:
        assert nchains_int<max_samples, "Attempted to draw %i samples from maximum of %i in nested_burnin" %(nchains_int, max_samples)
    else:
        for num_samples in nchains_int:
            assert nchains_int < max_samples, "Attempted to draw %i samples from maximum of %i in nested_burnin" % (
            num_samples, max_samples)

    #----

    if Nbands == 1:
        model = model_cont
    elif Nbands ==2:
        model = model_1line
    elif Nbands ==3:
        model = model_2line
    else:
        assert False, "nested burn in only current implemented for up to two lines"

    ns = NestedSampler(model, constructor_kwargs={"num_live_points": num_live_points,
                                                  "max_samples": max_samples})
    ns.run(jax.random.PRNGKey(seed), data)

    if type(nchains_int) == int:
        samples = ns.get_samples(jax.random.PRNGKey(seed), nchains_int)
    else:
        samples = [ns.get_samples(jax.random.PRNGKey(seed), nchains_int) for num_samples in nchains]

    return(samples)

def nested_transform(samples, to_nline = False):
    '''
    Transforms the samples from nested_burnin to the constrained domain. Returns as dictionary
    '''
    Nbands = int(len(samples.keys())/3)
    #----
    if Nbands == 1:
        transforms = {
            'log_sigma_c': numpyro.distributions.biject_to(numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max).support),
            'log_tau': numpyro.distributions.biject_to(numpyro.distributions.Uniform(log_tau_min, log_tau_max).support),
            'means_0': numpyro.distributions.biject_to(numpyro.distributions.Uniform(mean_min, mean_max).support),
        }

    elif Nbands ==2:
        transforms = {
            'log_sigma_c': numpyro.distributions.biject_to(numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max).support),
            'log_tau': numpyro.distributions.biject_to(numpyro.distributions.Uniform(log_tau_min, log_tau_max).support),
            'means_0': numpyro.distributions.biject_to(numpyro.distributions.Uniform(mean_min, mean_max).support),

            'lags_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(lag_min, lag_max).support),
            'rel_amps_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(rel_amp_min, rel_amp_max).support),
            'means_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(mean_min, mean_max).support),
        }
    elif Nbands ==3:
        transforms = {
            'log_sigma_c': numpyro.distributions.biject_to(numpyro.distributions.Uniform(log_sigma_c_min, log_sigma_c_max).support),
            'log_tau': numpyro.distributions.biject_to(numpyro.distributions.Uniform(log_tau_min, log_tau_max).support),
            'means_0': numpyro.distributions.biject_to(numpyro.distributions.Uniform(mean_min, mean_max).support),

            'lags_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(lag_min, lag_max).support),
            'rel_amps_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(rel_amp_min, rel_amp_max).support),
            'means_1': numpyro.distributions.biject_to(numpyro.distributions.Uniform(mean_min, mean_max).support),

            'lags_2': numpyro.distributions.biject_to(numpyro.distributions.Uniform(lag_min, lag_max).support),
            'rel_amps_2': numpyro.distributions.biject_to(numpyro.distributions.Uniform(rel_amp_min, rel_amp_max).support),
            'means_2': numpyro.distributions.biject_to(numpyro.distributions.Uniform(mean_min, mean_max).support),
        }
    else:
        assert False, "nested burn in only current implemented for up to two lines"

    for key in transforms.keys():
        assert key in samples.keys(), "bad key: %s\t in nested_transform" %key
    for key in samples.keys():
        assert key in transforms.keys(), "bad key: %s\t in nested_transform" %key

    #--------
    #Transform samples to constrained domain
    start_positions = transform_fn(transforms, samples, invert=True)
    # --------

    if to_nline == False:
        return start_positions
    else:
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
if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict

    print("Starting tests for fitting procedure")


    # load some example data
    rootfol = "./Data/data_fake/150day-bad/"

    truelag1=150
    truelag2=150

    cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
    line1 = array_to_lc(np.loadtxt(rootfol + "line1.dat"))
    line2 = array_to_lc(np.loadtxt(rootfol + "line2.dat"))

    #Make into banded format
    banded_2line  = lc_to_banded([cont, line1, line2])
    banded_1line  = lc_to_banded([cont, line1])
    banded_cont   = lc_to_banded([cont])

    datas = [banded_cont, banded_1line,banded_2line]
    models = [model_cont, model_1line, model_2line]

    '''
    for i in [2,1,0]:
        data = datas[i]
        print("\t Unit tests for %i lines: " %max(data["bands"]) )
        nested_burnin_tformed(data, nchains=10, num_live_points=10, max_samples=20)
    '''
    print("Unit tests succesfful")


    #==================================================================
    from chainconsumer import ChainConsumer
    import matplotlib.pylab as plt

    #---------------------------------------------
    for i in [1]:
        data = datas[i]
        nbands = max(data["bands"])+1
        nmodes = 1 + ((nbands-1)*2)**2
        nsamples = 4000*nmodes

        nest_results = nested_burnin(data, nchains=nsamples, seed = 1)

    #---------------------------------------------
    to_nline = False
    if to_nline:
        from modelling import model_nline
        model = model_nline
    else:
        model = models[i]
    #---------------------------------------------

    print("Doing transforms")
    nest_starts = nested_transform(nest_results, to_nline=to_nline)
    print("UNTRANSFORMED RESULTS")
    for key in nest_results.keys():
        print(key)
        print(nest_results[key])

    print("TRANSFORMED RESULTS")
    for key in nest_starts.keys():
        print(key)
        print(nest_starts[key])
    print("Making NUTS sampler")

    #---------------------------------------------
    '''
    sampler = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model = model, step_size=0.001, target_accept_prob=0.9),
        num_warmup= 200,
        num_chains=nsamples,
        num_samples= 200,
    )

    print("Running sampler")
    sampler.run(jax.random.PRNGKey(0), data, init_params=nest_starts)
    print("Getting samples and plotting")
    NUTS_samples = sampler.get_samples()
    '''

    #---------------------------------------------

    c= ChainConsumer()
    c.add_chain(nest_results)
    c.plotter.plot(extents=plot_extents)
    plt.tight_layout()
    plt.title("Nested Sampling Results")
    plt.show()

    '''
    c2= ChainConsumer()
    if to_nline:
        NUTS_samples = dict(NUTS_samples)
        print(NUTS_samples)
        NUTS_samples,keys= flatten_dict(NUTS_samples)
        print(NUTS_samples)
        c2.add_chain(NUTS_samples , parameters=keys)
    else:
        c2.add_chain(NUTS_samples)

    for key in NUTS_samples.keys():
        print(key)
        print(NUTS_samples[key])

    c2.plotter.plot(extents=plot_extents)
    plt.tight_layout()
    plt.title("NUTS results")
    plt.show()
    '''