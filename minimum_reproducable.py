'''
An attempt at creating a minimum reproducable example of a model fit

V1 - Continuum recovery
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

#==================================================

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


def time_delay_transform(lag, X):
    t, band = X
    return t - lag[band]



#==================================================

def model(data):

    T = jnp.array(data['T'])
    Y = jnp.array(data['Y'])
    E = jnp.array(data['E'])

    bands = jnp.array(data['bands'])

    # ===================

    log_tau     = numpyro.sample('log_tau',     numpyro.distributions.Uniform(1,8))
    #log_tau = jnp.log(400)
    log_sigma   = numpyro.sample('log_sigma',   numpyro.distributions.Uniform(-4,4))
    rel_amp     = numpyro.sample('rel_amp',     numpyro.distributions.Uniform(0,10))
    mean        = numpyro.sample('mean',        numpyro.distributions.Uniform(-10,10))
    lag         = numpyro.sample('lag',         numpyro.distributions.Uniform(0,1000))

    # ===================

    tau     = jnp.exp(log_tau)
    sigma   = jnp.exp(log_sigma)
    lags    = jnp.array([0, lag])

    # ===================
    T -=lags[bands]
    Y -= mean

    inds = jnp.argsort(T)

    T = T[inds]
    Y = Y[inds]
    E = E[inds]
    bands = bands[inds]


    gp = build_gp(T, E, tau, sigma, 0.0, rel_amp, bands)

    numpyro.sample('y', gp.numpyro_dist(), obs = Y)


def build_gp(T, E, tau, sigma, mean, rel_amp, bands):

    base_kernel = tinygp.kernels.quasisep.Matern32(scale = tau)

    kernel = Multiband(
        amplitudes  =   jnp.array([1,rel_amp])*sigma,
        kernel  =   base_kernel,
    )

    meanf = lambda t: mean

    out = tinygp.GaussianProcess(kernel, (T,bands), diag=E*E)

    return(out)

@jax.jit
def loss(lag,log_tau):

    T = jnp.array(data['T'])
    Y = jnp.array(data['Y'])
    E = jnp.array(data['E'])

    bands = jnp.array(data['bands'])

    # ===================

    sigma       = 1
    rel_amp     = 1
    mean        = 0


    # ===================

    tau     = jnp.exp(log_tau)
    lags    = jnp.array([0, lag])

    # ===================
    T = time_delay_transform(lags, (T,bands))
    Y -= mean

    inds = jnp.argsort(T)

    T = T[inds]
    Y = Y[inds]
    E = E[inds]
    bands = bands[inds]

    gp = build_gp(T, E, tau, sigma, mean, rel_amp, bands)

    return gp.log_probability(Y)


#=======================

if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict
    #load some example data

    cont  = array_to_lc(np.loadtxt("./Data/data_fake/360day/cont.dat"))
    line1 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line1.dat"))

    #Make into banded format
    data = lc_to_banded([cont, line1])
    rel_amp = 1
    lag = 360

    #================================

    true_params={
        'log_tau': jnp.log(400),
        'log_sigma': 0,
        'rel_amp': rel_amp,
        'mean': 0,
        'lag': lag
    }

    extents = {
        'log_tau': [1,8],
        'log_sigma': [-4,4],
        'rel_amp': [0,10],
        'mean': [-10, 10],
        'lag': [0,1000]
    }

    init_params = copy(true_params)
    init_params.pop('lag')

    nchains     = 40
    nburn       = 200
    nsample     = 600

    #Create and run sampler

    sampler = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
                            model,
                           init_strategy=numpyro.infer.init_to_value(values=init_params),
                            target_accept_prob=0.2

                           ),
        num_chains = nchains,
        num_warmup = nburn,
        num_samples = nsample,
        )


    sampler.run(jax.random.PRNGKey(1), data)

    out=sampler.get_samples()
    print("sampling done")

    #------------------------------
    c = ChainConsumer()
    c.add_chain(out)
    c.plotter.plot(truth=true_params, extents=extents)
    plt.tight_layout()
    plt.show()

    #------------------------------
    from data_utils import banded_to_lc
    fig, axs = plt.subplots(2,1)
    lc1,lc2 = banded_to_lc(data)
    plag = 0
    for ax in axs:
        ax.axhline(0, ls='--', c='k',alpha=0.5,lw=2)

        ax.errorbar(lc1['T'],lc1['Y'],yerr = lc1['E'], fmt="none", c='b')
        ax.errorbar(lc2['T']-plag,lc2['Y']/rel_amp,yerr = lc2['E']/rel_amp, fmt="none", c='r')
        ax.set_ylim([-3,3])

        plag = lag
    plt.show()

    #------------------------------
    nplot=128
    lag_plot    = np.linspace(0, 1000, nplot)
    logtau_plot = np.linspace(1, 8, nplot)
    X, Y = jnp.meshgrid(lag_plot, logtau_plot)
    Z=np.zeros([nplot,nplot])
    for i in range(nplot):
        if i % 50 == 0: print(i)
        for j in range(nplot):
            lag     =lag_plot[i]
            log_tau = logtau_plot[j]
            Z[i,j] = loss(lag,log_tau)
    Z=Z.T


    print('grid done, plotting')


    plt.figure()
    plt.imshow(Z[::-1, ::],
               interpolation='none', cmap='viridis',
               extent=[min(lag_plot),max(lag_plot),min(logtau_plot),max(logtau_plot)],
               aspect="auto")
    plt.axvline(true_params['lag'])
    plt.axhline(true_params['log_tau'])
    plt.title("log probability")
    plt.xlabel("lag")
    plt.ylabel("log_tau")
    plt.show()

    #------------------------------
    plt.figure()
    Z2 = np.log(np.exp(Z))
    plt.imshow(Z2[::-1, ::],
               interpolation='none', cmap='viridis',
               extent=[min(lag_plot),max(lag_plot),min(logtau_plot),max(logtau_plot)],
               aspect="auto")
    plt.axvline(true_params['lag'])
    plt.axhline(true_params['log_tau'])
    plt.title("log probability, filtered")
    plt.xlabel("lag")
    plt.ylabel("log_tau")
    plt.show()

    #------------------------------
    plt.figure()
    X = out['lag']
    Y = out['log_tau']

    for i in range(nchains):
        if i%50==0: print(i)
        plt.plot(X[i*nsample:(i+1)*nsample], Y[i*nsample:(i+1)*nsample], lw=0.1)

    plt.axvline(true_params['lag'])
    plt.axhline(true_params['log_tau'])
    plt.xlabel("lag")
    plt.ylabel("log_tau")
    plt.title("Chain paths")
    plt.show()