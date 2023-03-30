import matplotlib.pyplot as plt

import warnings
import os

from chainconsumer import ChainConsumer

import jax
import jax.numpy as jnp
#import jaxopt
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from tinygp import GaussianProcess, kernels, transforms
import tinygp
import numpy as np

import matplotlib as mpl
from functools import partial

#============================================
warnings.filterwarnings("ignore", category=FutureWarning)
numpyro.set_host_device_count(6)

#============================================
#Utility Funcs
def mean_func(means, X):
    '''
    Utitlity function to take array of constants and return as gp-friendly functions
    '''
    t, band = X
    return(means[band])

def lc_to_banded(lcs):
    '''
    Takes a list of dicts {T, Y, E} of lightcurve objects and returns as single banded lightcurve
    '''
    Nbands = len(lcs)

    T = jnp.concatenate([lc['T'] for lc in lcs])
    Y = jnp.concatenate([lc['Y'] for lc in lcs])
    E = jnp.concatenate([lc['E'] for lc in lcs])

    bands = jnp.concatenate([jnp.zeros(len(lc['T']), dtype='int32') + band for band,lc in zip(range(Nbands),lcs)])

    out = {
        'T': T,
        'Y': Y,
        'E': E,
        'bands': bands,
    }

    return(out)

def banded_to_lc(data):
    '''
    Takes banded data and returns as a list of dicts with keys {T, Y, E}
    '''

    #Find how many bands there are
    bands = data['bands']
    Nbands = jnp.max(bands)+1
    out=[]

    #Split into dictionaries and append
    for i in range(Nbands):
        T = data['T'][bands==i]
        Y = data['Y'][bands==i]
        E = data['E'][bands==i]

        out.append({
            'T':T,
            'Y':Y,
            'E':E,
            })

    return(out)


def flatten_dict(dict):
    '''
    Unpacks all entries in a dictionary into a numpy  friendly array
    '''

    keys = list(dict.keys())
    sizes = [1]*len(keys)

    Ncols = 0
    Nrows = dict[keys[0]].shape[0]

    #Figure out how many cols we need
    for key,i in zip(keys,range(len(keys))):
        if len(dict[key].shape)>1: sizes[i]=dict[key].shape[1]
        assert dict[key].shape[0]==Nrows, "To flatten_dict, all entries must be same length"
        Ncols+=sizes[i]

    #Read data into numpy array
    out = np.zeros([Nrows,Ncols])
    out_keys = [''] * Ncols
    i=0

    for key,k in zip(keys,range(len(keys))):
        for j in range(sizes[k]):
            if sizes[k]>1:
                out[:, i] = dict[key][:,j]
                out_keys[i] = keys[k]+"_"+str(j)
            else:
                out[:, i] = dict[key][:]
                out_keys[i] = keys[k]
            i+=1

    return(out,out_keys)

#============================================
#Main Working Funcs
#============================================

#============================================
#TinyGP side

def build_gp(data, params, basekernel=tinygp.kernels.Exp):
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

    #Sort data into gp friendly format
    sort_inds = jnp.argsort(T)

    #Make GP
    kernel = basekernel(scale = tau)
    gp = GaussianProcess(
        kernel,
        T[sort_inds],
        diag=E[sort_inds]**2,
                         )

    out = (gp, sort_inds)
    return(out)


#============================================
#Numpyro Side

def model(data):
    '''
    Main model, to be fed to a numpyro NUTS object, with banded 'data' as an object
    '''
    #Continuum properties
    log_sigma   = numpyro.sample('log_sigma',   numpyro.distributions.Uniform(-5,5))
    log_tau     = numpyro.sample('log_tau',     numpyro.distributions.Uniform(0,10))

    #Find maximum number of bands in modelling
    Nbands = jnp.max(data['bands'])+1

    #Though we fit in logspace, we need non-log space properties for feeding to the GP
    cont_scale  = numpyro.deterministic('cont_scale',   jnp.exp(log_sigma))
    tau_d       = numpyro.deterministic('tau_d',        jnp.exp(log_tau))

    #Lag and scaling of respone lines
    lags = numpyro.sample('lags', numpyro.distributions.Uniform(0,  180*4),  sample_shape=(Nbands-1,))
    amps = numpyro.sample('amps', numpyro.distributions.Uniform(0,  100),    sample_shape=(Nbands-1,))

    #Means
    means = numpyro.sample('means', numpyro.distributions.Uniform(-100,100), sample_shape=(Nbands,))

    params = {
        'log_tau': log_tau,
        'log_sigma_c': log_sigma,
        'lags': lags,
        'amps': amps,
        'means': means,
    }

    #Build TinyGP Process
    gp, sort_inds = build_gp(data, params)

    #Apply likelihood
    numpyro.sample('y', gp.numpyro_dist(), obs=data['Y'][sort_inds])


#============================================
#Main Runtime Test
if __name__=="__main__":
    print("Running.")

    #Load in data
    cont_url  = "./Data/B1-2940510474_CIV/2940510474_CIV_exp.txt"
    line1_url = "./Data/B1-2940510474_CIV/2940510474_gBand.txt"
    line2_url = "./Data/B1-2940510474_CIV/2940510474_MgII.txt"

    #Read files and sort into banded form
    lcs = []
    for url in [cont_url,line1_url,line2_url]:
        data=np.loadtxt(url)
        lcs.append({
            "T": data[:,0],
            "Y": data[:,1]-np.min(data[:,1]),
            "E": data[:,2],
        })
    lcs = lc_to_banded(lcs)

    lcs['T']-=np.min(lcs['T'])

    out, out_keys = flatten_dict(lcs)
    np.savetxt("banded_data.dat", out)

    #Construct and run sampler
    sampler = numpyro.infer.MCMC(infer.NUTS(model), num_chains=300, num_warmup=200, num_samples=600)
    sampler.run(jax.random.PRNGKey(0),lcs)
    output = sampler.get_samples()
    output.pop('means')

    out,out_keys = flatten_dict(output)

    np.savetxt("./Data/B1-2940510474_CIV/outchain.dat",out)
    np.savetxt("./Data/B1-2940510474_CIV/outchain_keys.dat",out_keys,fmt="%s")
