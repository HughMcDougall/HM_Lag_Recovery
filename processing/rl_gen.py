'''
Used for creating R-L relationship type graphs from our data
'''

import warnings

import data_utils

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import sys
import pandas as pd
from copy import deepcopy as copy
from jax.random import PRNGKey

import numpyro
from numpyro_ext.distributions import MixtureGeneral
import jax.numpy as jnp

import matplotlib.pylab as plt
from chainconsumer import ChainConsumer

from scipy.signal import argrelextrema
import data_utils


sys.path.append("..")

#------------------------------------
def getpeak(data):
    h = np.histogram(data, bins = 64)
    hData = h[0]
    i = np.where(hData == np.max(hData))
    return(np.mean(h[1][i]))
#------------------------------------
CASE = "MGII"

LOGLUM        = np.loadtxt("./results/"+CASE+"_lums.dat")
Z             = np.loadtxt("./results/"+CASE+"_zs.dat")
LAGS_IND      = np.loadtxt("./results/"+CASE+"_samples_ind.dat")
LAGS_SIM      = np.loadtxt("./results/"+CASE+"_samples_sim.dat")

rootfol="../"
N = LAGS_SIM.shape[1]

ISGOOD_IND = [True]*N
ISGOOD_SIM = [True]*N
#------------------------------------
#Figure 1: Plot loglog with scatterplot of lags
#Todo - span this out to all 4 lag types
fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

alpha = 1/255
size = 5
sparseness = 4
Lscatter = 0.005

X_ind_formodel = []
Y_ind_formodel = []

X_sim_formodel = []
Y_sim_formodel = []

print("Doing plots")
for i in range(N):
    print("\t %i" %(i))

    L = LOGLUM[i]
    z = Z[i]

    Y_ind = np.log10(LAGS_IND[:,i]/(1+z))
    Y_sim = np.log10(LAGS_SIM[:,i]/(1+z))

    X_ind = np.ones_like(Y_ind) * L + np.random.randn(len(Y_ind))*Lscatter
    X_sim = np.ones_like(Y_sim) * L + np.random.randn(len(Y_sim))*Lscatter


    sig_width_ind       = abs(np.percentile(LAGS_IND[:,i],84.13 )-np.percentile(LAGS_IND[:,i],15.87))
    med_mode_diff_ind   = abs(np.median(LAGS_IND[:,i]) - getpeak(LAGS_IND[:,i]))

    sig_width_sim       = abs(np.percentile(LAGS_SIM[:,i],84.13 )-np.percentile(LAGS_SIM[:,i],15.87))
    med_mode_diff_sim   = abs(np.median(LAGS_SIM[:,i]) - getpeak(LAGS_SIM[:,i]))

    if  sig_width_ind> 100 or med_mode_diff_ind>30: ISGOOD_IND[i]=False
    if  sig_width_sim> 100 or med_mode_diff_sim>30: ISGOOD_SIM[i]=False

    if ISGOOD_IND:
        ax[0].scatter(X_ind[::sparseness], Y_ind[::sparseness], alpha=alpha, s = size, c='blue')
        X_ind_formodel.append(X_ind)
        Y_ind_formodel.append(Y_ind)
    if ISGOOD_SIM:
        ax[1].scatter(X_sim[::sparseness], Y_sim[::sparseness], alpha=alpha, s = size, c='blue')
        X_sim_formodel.append(X_sim)
        Y_sim_formodel.append(Y_sim)

print("If %i measurements for %s, %i remain for independent fits and %i for simultaneous" %(N,CASE,sum(ISGOOD_IND),sum(ISGOOD_SIM)))
ax[0].set_xlim(44.2, 46.5)
ax[0].set_ylim(np.log10(1),np.log10(3200))
fig.supxlabel("$log_{10}(\lambda L _{3000})$")
fig.supylabel("$log_{10} ( (1+z)^{-1}  \Delta t_{%s} )$" %CASE)


fig.suptitle("%s Lag R-L Plot, After Quality Cut" %CASE)
fig.savefig(fname="./RL-"+CASE+".png", format='png')

#-----------------------------------------------

X_ind_formodel = jnp.concatenate(X_ind_formodel)
Y_ind_formodel = jnp.concatenate(Y_ind_formodel)

X_sim_formodel = jnp.concatenate(X_sim_formodel)
Y_sim_formodel = jnp.concatenate(Y_sim_formodel)

#-----------------------------------------------
def linear_mixture_model(x, y):
    m   = numpyro.sample("m", numpyro.distributions.Uniform(-2,2))
    b   = numpyro.sample("b", numpyro.distributions.Uniform(-50,20))
    dex =numpyro.sample("dex", numpyro.distributions.Uniform(0,10))

    fg_dist = numpyro.distributions.Normal(m * x + b, dex)

    bg_mean = numpyro.sample("bg_mean", numpyro.distributions.Normal(0.0, 10))
    bg_sigma = numpyro.sample("bg_sigma", numpyro.distributions.HalfNormal(10))
    bg_dist = numpyro.distributions.Normal(bg_mean, bg_sigma)


    Q = numpyro.sample("Q", numpyro.distributions.Uniform(0.0, 1.0))
    mix = numpyro.distributions.Categorical(probs=jnp.array([Q, 1.0 - Q]))


    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", MixtureGeneral(mix, [fg_dist, bg_dist]), obs=y)

def linear_model_standard(x, y):
    m   = numpyro.sample("m", numpyro.distributions.Uniform(-2,2))
    b   = numpyro.sample("b", numpyro.distributions.Uniform(-50,20))
    dex =numpyro.sample("dex", numpyro.distributions.Uniform(0,20))

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", numpyro.distributions.Normal(m*x+b, dex), obs=y)
#=========

nburn    = 400
nsamples = 300
nchains  = 400
nbins    = None

do_simp = True
do_mix = True

CALC  = [False,False,False,True]

#=========
# DO NUMPYRO RUNS, GET SAMPLES ETC
if do_simp:
    if CALC[0]:
        sampler_ind_simple = numpyro.infer.MCMC(
            numpyro.infer.NUTS(linear_model_standard),
            num_warmup=nburn,
            num_samples=nsamples,
            num_chains=nchains,
            progress_bar=True,
        )
        sampler_ind_simple.run(PRNGKey(3), x=X_ind_formodel, y=Y_ind_formodel)

        samples_ind_simple = sampler_ind_simple.get_samples()
        a, b = data_utils.flatten_dict(samples_ind_simple)
        np.savetxt(fname="./%s_RLchain_ind_simple.dat" % CASE, X=a, delimiter='\t')
        del a,b, samples_ind_simple
    else:
        a = np.loadtxt(fname="./%s_RLchain_ind_simple.dat" % CASE, delimiter='\t')
        samples_ind_simple = {'b':      a[:,0],
                              'm':      a[:,1],
                              'dex':    a[:,2]
                              }
        del a

    #=========
    if CALC[1]:
        sampler_sim_simple = numpyro.infer.MCMC(
            numpyro.infer.NUTS(linear_model_standard),
            num_warmup=nburn,
            num_samples=nsamples,
            num_chains=nchains,
            progress_bar=True,
        )
        sampler_sim_simple.run(PRNGKey(3), x=X_sim_formodel, y=Y_sim_formodel)

        samples_sim_simple = sampler_sim_simple.get_samples()
        a, b = data_utils.flatten_dict(samples_sim_simple)
        np.savetxt(fname="./%s_RLchain_sim_simple.dat" % CASE, X=a, delimiter='\t')
        del a, b, sampler_sim_simple
    else:
        a = np.loadtxt(fname="./%s_RLchain_sim_simple.dat" % CASE, delimiter='\t')
        samples_ind_simple = {'b':      a[:,0],
                              'm':      a[:,1],
                              'dex':    a[:,2]
                              }
        del a


#=========
if do_mix:

    if CALC[2]:
        sampler_ind_mix = numpyro.infer.MCMC(
            numpyro.infer.NUTS(linear_mixture_model),
            num_warmup=nburn,
            num_samples=nsamples,
            num_chains=nchains,
            progress_bar=True,
        )
        sampler_ind_mix.run(PRNGKey(3), x=X_ind_formodel, y=Y_ind_formodel)

        samples_ind_mix     = sampler_ind_mix.get_samples()
        a, b = data_utils.flatten_dict(samples_ind_mix)
        np.savetxt(fname="./%s_RLchain_ind_mix.dat" % CASE, X=a, delimiter='\t')
        del a, b, sampler_ind_mix
    else:
        a = np.loadtxt(fname="./%s_RLchain_ind_mix.dat" % CASE, delimiter='\t')
        samples_ind_simple = {'Q':      a[:,0],
                              'b':      a[:,1],
                              'bg_mean':    a[:,2],
                              'bg_sigma': a[:, 3],
                              'dex': a[:, 4],
                              'm': a[:, 5]
                              }
        del a


    #=========
    if CALC[3]:
        sampler_sim_mix = numpyro.infer.MCMC(
            numpyro.infer.NUTS(linear_mixture_model),
            num_warmup=nburn,
            num_samples=nsamples,
            num_chains=nchains,
            progress_bar=True,
        )
        sampler_sim_mix.run(PRNGKey(3), x=X_sim_formodel, y=Y_sim_formodel)

        samples_sim_mix = sampler_sim_mix.get_samples()

        a, b = data_utils.flatten_dict(samples_sim_mix)
        np.savetxt(fname="./%s_RLchain_sim_mix.dat" % CASE, X=a, delimiter='\t')
        del a,b, sampler_sim_mix
    else:
        a = np.loadtxt(fname="./%s_RLchain_sim_mix.dat" % CASE, delimiter='\t')
        samples_ind_simple = {'Q':      a[:,0],
                              'b':      a[:,1],
                              'bg_mean':    a[:,2],
                              'bg_sigma': a[:, 3],
                              'dex': a[:, 4],
                              'm': a[:, 5]
                              }
        del a


    #=========


#---------------------------------------------------------
# DO CHAINCONSUMER PLOTS


if do_simp:
    c = ChainConsumer()
    # ==========
    #Add chains with renamed vars

    c.add_chain({'R-L Slope': samples_ind_simple['m'],
                 'R-L Offset':samples_ind_simple['b']},
                  name="Independent, Simple")
    c.add_chain({'R-L Slope': samples_sim_simple['m'],
                 'R-L Offset':samples_sim_simple['b']},
                  name="Simultaneous, Simple")

    # ==========
    # Configure & Plot
    c.configure(
            legend_kwargs={"loc": "lower left", "fontsize": 10},
            legend_color_text=True,
            bins = nbins,
            )
    cfig = c.plotter.plot(
                   chains=["Independent, Simple", "Simultaneous, Simple"],
                   extents={'R-L Slope':[-1,1],
                            'R-L Offset':[-50,20]})

    cfig.set_size_inches(4,4)
    cfig.tight_layout()
    cfig.savefig("%s_R-L-Params_simple.png" %CASE)

    del c

if do_mix:
    c = ChainConsumer()
    # ==========
    #Add chains with renamed vars

    c.add_chain({'R-L Slope': samples_ind_mix['m'],
                 'R-L Offset':samples_ind_mix['b']},
                  name="Independent, Mix")
    c.add_chain({'R-L Slope': samples_sim_mix['m'],
                 'R-L Offset':samples_sim_mix['b']},
                  name="Simultaneous, Mix")

    # ==========
    # Configure & Plot
    c.configure(
            legend_kwargs={"loc": "lower left", "fontsize": 10},
            legend_color_text=True,
            bins = nbins,
            )
    cfig=c.plotter.plot(
                   chains=["Independent, Mix", "Simultaneous, Mix"],
                   extents={'R-L Slope':[-1,1],
                            'R-L Offset':[-50,20]})
    cfig.set_size_inches(4,4)
    cfig.tight_layout()
    cfig.savefig("%s_R-L-Params_mix.png" %CASE)

    del c

#---------------------------------------------------------
# PLOT RECOVRED R-L'S ON SCATTERPLOT


Lplot = np.linspace(44.2, 46.5)
if do_simp:
    #   Get median values from MCMC runs
    b_sim_simple = np.median(samples_sim_simple['b'])
    m_sim_simple = np.median(samples_sim_simple['m'])
    dex_sim_simple = np.median(samples_sim_simple['dex'])

    b_ind_simple = np.median(samples_ind_simple['b'])
    m_ind_simple = np.median(samples_ind_simple['m'])
    dex_ind_simple = np.median(samples_ind_simple['dex'])
    # ============

    #   Overlay on scatterplot

    ax[0].plot(Lplot, Lplot * m_ind_simple + b_ind_simple, c='r', ls='-', lw=1,
               label='Simple Regression (Median Values)')
    ax[0].plot(Lplot, Lplot * m_ind_simple + b_ind_simple + dex_ind_simple, c='r', ls='--', lw=1)
    ax[0].plot(Lplot, Lplot * m_ind_simple + b_ind_simple - dex_ind_simple, c='r', ls='--', lw=1)

    ax[1].plot(Lplot, Lplot * m_ind_simple + b_ind_simple, c='r', ls='-', lw=1,
               label='Simple Regression (Median Values)')
    ax[1].plot(Lplot, Lplot * m_ind_simple + b_ind_simple + dex_ind_simple, c='r', ls='--', lw=1)
    ax[1].plot(Lplot, Lplot * m_ind_simple + b_ind_simple - dex_ind_simple, c='r', ls='--', lw=1)

if do_mix:
    #   Get median values from MCMC runs
    b_sim_mix = np.median(samples_sim_mix['b'])
    m_sim_mix = np.median(samples_sim_mix['m'])
    dex_sim_mix = np.median(samples_sim_mix['dex'])

    b_ind_mix = np.median(samples_ind_mix['b'])
    m_ind_mix = np.median(samples_ind_mix['m'])
    dex_ind_mix = np.median(samples_ind_mix['dex'])

    #============

    #   Overlay on scatterplot

    ax[0].plot(Lplot, Lplot * m_ind_mix + b_ind_mix, c='b', ls='-', lw=1,
               label='Mixture Model (Median Values)')
    ax[0].plot(Lplot, Lplot * m_ind_mix + b_ind_mix + dex_ind_mix, c='b', ls='--', lw=1)
    ax[0].plot(Lplot, Lplot * m_ind_mix + b_ind_mix - dex_ind_mix, c='b', ls='--', lw=1)

    ax[1].plot(Lplot, Lplot * m_ind_mix + b_ind_mix, c='b', ls='-', lw=1,
               label='Mixture Model (Median Values)')
    ax[1].plot(Lplot, Lplot * m_ind_mix + b_ind_mix + dex_ind_mix, c='b', ls='--', lw=1)
    ax[1].plot(Lplot, Lplot * m_ind_mix + b_ind_mix - dex_ind_mix, c='b', ls='--', lw=1)

fig.savefig(fname="./RL-"+CASE+"with-regs.png", format='png')

#---------------------------------------------------------

plt.show()