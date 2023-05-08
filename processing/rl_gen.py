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
fig_nocut, ax_nocut = plt.subplots(1,2, sharex=True, sharey=True)

alpha = 1/255
size = 5
sparseness = 4

Lscatter = 0.005

X_ind_formodel = []
Y_ind_formodel = []
E_ind_formodel = []

X_sim_formodel = []
Y_sim_formodel = []
E_sim_formodel = []

print("Doing plots")
for i in range(N):
    print("\t %i" %(i))

    #Get data and shift into rest frame
    L = LOGLUM[i]
    z = Z[i]

    Y_ind = np.log10(LAGS_IND[:,i]/(1+z))
    Y_sim = np.log10(LAGS_SIM[:,i]/(1+z))

    #Add scatter in x dim to make plots nicer
    X_ind = np.ones_like(Y_ind) * L + np.random.randn(len(Y_ind))*Lscatter
    X_sim = np.ones_like(Y_sim) * L + np.random.randn(len(Y_sim))*Lscatter

    #Determine width and median - mode difference for all datasets
    sig_width_ind       = abs(np.percentile(LAGS_IND[:,i],84.13 )-np.percentile(LAGS_IND[:,i],15.87)) / 2
    med_mode_diff_ind   = abs(np.median(LAGS_IND[:,i]) - getpeak(LAGS_IND[:,i]))

    sig_width_sim       = abs(np.percentile(LAGS_SIM[:,i],84.13 )-np.percentile(LAGS_SIM[:,i],15.87)) / 2
    med_mode_diff_sim   = abs(np.median(LAGS_SIM[:,i]) - getpeak(LAGS_SIM[:,i]))

    # Perform quality cuts
    if  sig_width_ind> 40 or med_mode_diff_ind>100: ISGOOD_IND[i]=False
    if  sig_width_sim> 40 or med_mode_diff_sim>100: ISGOOD_SIM[i]=False

    ax_nocut[0].scatter(X_ind[::sparseness], Y_ind[::sparseness], alpha=alpha, s = size, c='blue')
    ax_nocut[1].scatter(X_sim[::sparseness], Y_sim[::sparseness], alpha=alpha, s=size, c='blue')
    if ISGOOD_IND[i]:
        ax[0].scatter(X_ind[::sparseness], Y_ind[::sparseness], alpha=alpha, s = size, c='blue')
        X_ind_formodel.append(L)
        Y_ind_formodel.append(np.median(Y_ind))
        E_ind_formodel.append((np.percentile(Y_ind,84.13)-np.percentile(Y_ind,15.87))/ 2)

    if ISGOOD_SIM[i]:
        ax[1].scatter(X_sim[::sparseness], Y_sim[::sparseness], alpha=alpha, s = size, c='blue')
        X_sim_formodel.append(L)
        Y_sim_formodel.append(np.median(Y_sim))
        E_sim_formodel.append((np.percentile(Y_sim,84.13)-np.percentile(Y_sim,15.87))/ 2)

print("If %i measurements for %s, %i remain for independent fits and %i for simultaneous" %(N,CASE,sum(ISGOOD_IND),sum(ISGOOD_SIM)))
for figi, axi in zip([fig,fig_nocut],[ax,ax_nocut]):
    axi[0].set_xlim(44.2, 46.5)
    axi[0].set_ylim(np.log10(1),np.log10(3200))
    figi.supxlabel("$log_{10}(\lambda L _{3000})$")
    figi.supylabel("$log_{10} ( (1+z)^{-1}  \Delta t_{%s} )$" %CASE)


fig.suptitle("%s Lag R-L Plot, After Quality Cut" %CASE)
fig.savefig(fname="./RL-"+CASE+".png", format='png')
fig_nocut.suptitle("%s Lag R-L Plot, Before Quality Cut" %CASE)
fig_nocut.savefig(fname="./RL-"+CASE+"-nocut.png", format='png')

#-----------------------------------------------
def linear_mixture_model(x, y, e):
    m   = numpyro.sample("m", numpyro.distributions.Uniform(-2,2))
    b   = numpyro.sample("b", numpyro.distributions.Uniform(-50,20))
    dex =numpyro.sample("dex", numpyro.distributions.Uniform(0,10))

    fg_dist = numpyro.distributions.Normal(m * x + b, jnp.sqrt(dex**2+e**2))

    bg_mean = numpyro.sample("bg_mean", numpyro.distributions.Normal(0.0, 10))
    bg_sigma = numpyro.sample("bg_sigma", numpyro.distributions.HalfNormal(10))
    bg_dist = numpyro.distributions.Normal(bg_mean, jnp.sqrt(bg_sigma**2+e**2))


    Q = numpyro.sample("Q", numpyro.distributions.Uniform(0.0, 1.0))
    mix = numpyro.distributions.Categorical(probs=jnp.array([Q, 1.0 - Q]))


    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", MixtureGeneral(mix, [fg_dist, bg_dist]), obs=y)

def linear_model_standard(x, y, e):
    m   = numpyro.sample("m", numpyro.distributions.Uniform(-4,4))
    b   = numpyro.sample("b", numpyro.distributions.Uniform(-200,40))
    dex =numpyro.sample("dex", numpyro.distributions.Uniform(0,20))

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", numpyro.distributions.Normal(m*x+b, jnp.sqrt(dex**2+e**2)), obs=y)

#----------------------------------------------
E_ind_formodel = jnp.array(E_ind_formodel )
X_ind_formodel = jnp.array(X_ind_formodel )
Y_ind_formodel = jnp.array(Y_ind_formodel)

E_sim_formodel = jnp.array(E_sim_formodel )
X_sim_formodel = jnp.array(X_sim_formodel )
Y_sim_formodel = jnp.array(Y_sim_formodel )


#----------------------------------------------
fig2, ax2 = plt.subplots(1,2, sharex=True, sharey=True)
for x,y,e in zip(X_ind_formodel,Y_ind_formodel,E_ind_formodel):
    ax2[0].errorbar(x,y,yerr=e,fmt='none',alpha=0.1, c='b')
    ax2[0].scatter(x, y, alpha=0.5, c='b', s=2)
for x,y,e in zip(X_sim_formodel,Y_sim_formodel,E_sim_formodel):
    ax2[1].errorbar(x,y,yerr=e,fmt='none',alpha=0.1, c='b')
    ax2[1].scatter(x, y, alpha=0.5, c='b', s=2)

ax2[0].set_xlim(44.2, 46.5)
ax2[0].set_ylim(np.log10(1),np.log10(3200))

fig2.supxlabel("$log_{10}(\lambda L _{3000})$")
fig2.supylabel("$log_{10} ( (1+z)^{-1}  \Delta t_{%s} )$" %CASE)


fig2.suptitle("%s Lag R-L Plot, After Quality Cut, Summarized" %CASE)

#=========

nburn    = 1000
nsamples = 1000
nchains  = 20

nbins    = None

do_simp = True
do_mix  = False

CALC  = [True, True, False, False]

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
        sampler_ind_simple.run(PRNGKey(3), x=X_ind_formodel, y=Y_ind_formodel, e = E_ind_formodel)

        samples_ind_simple = sampler_ind_simple.get_samples()
        a, b = data_utils.flatten_dict(samples_ind_simple)
        np.savetxt(fname="./%s_RLchain_ind_simple.dat" % CASE, X=a, delimiter='\t')
        del a,b, sampler_ind_simple
    else:
        a = np.loadtxt(fname="./%s_RLchain_ind_simple.dat" % CASE, delimiter='\t')
        samples_ind_simple = {'b':      a[:,0],
                              'dex':      a[:,1],
                              'm':    a[:,2]
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
        sampler_sim_simple.run(PRNGKey(3), x=X_sim_formodel, y=Y_sim_formodel, e = E_sim_formodel)

        samples_sim_simple = sampler_sim_simple.get_samples()
        a, b = data_utils.flatten_dict(samples_sim_simple)
        np.savetxt(fname="./%s_RLchain_sim_simple.dat" % CASE, X=a, delimiter='\t')
        del a, b, sampler_sim_simple
    else:
        a = np.loadtxt(fname="./%s_RLchain_sim_simple.dat" % CASE, delimiter='\t')
        samples_sim_simple = {'b':      a[:,0],
                              'dex':      a[:,1],
                              'm':    a[:,2]
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
        sampler_ind_mix.run(PRNGKey(3), x=X_ind_formodel, y=Y_ind_formodel, e = E_ind_formodel)

        samples_ind_mix     = sampler_ind_mix.get_samples()
        a, b = data_utils.flatten_dict(samples_ind_mix)
        np.savetxt(fname="./%s_RLchain_ind_mix.dat" % CASE, X=a, delimiter='\t')
        del a, b, sampler_ind_mix
    else:
        a = np.loadtxt(fname="./%s_RLchain_ind_mix.dat" % CASE, delimiter='\t')
        samples_ind_mix = {'Q':      a[:,0],
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
        sampler_sim_mix.run(PRNGKey(3), x=X_sim_formodel, y=Y_sim_formodel, e = E_sim_formodel)

        samples_sim_mix = sampler_sim_mix.get_samples()

        a, b = data_utils.flatten_dict(samples_sim_mix)
        np.savetxt(fname="./%s_RLchain_sim_mix.dat" % CASE, X=a, delimiter='\t')
        del a,b, sampler_sim_mix
    else:
        a = np.loadtxt(fname="./%s_RLchain_sim_mix.dat" % CASE, delimiter='\t')
        samples_sim_mix = {'Q':      a[:,0],
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
k = 1
noreals = int(128/k)
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
    ax2[0].plot(Lplot, Lplot * m_ind_simple + b_ind_simple, c='r', ls='-', lw=1,
               label='Simple Regression (Median Values)')
    ax2[0].plot(Lplot, Lplot * m_ind_simple + b_ind_simple + dex_ind_simple, c='r', ls='--', lw=1)
    ax2[0].plot(Lplot, Lplot * m_ind_simple + b_ind_simple - dex_ind_simple, c='r', ls='--', lw=1)


    ax2[1].plot(Lplot, Lplot * m_sim_simple + b_sim_simple, c='r', ls='-', lw=1,
               label='Simple Regression (Median Values)')
    ax2[1].plot(Lplot, Lplot * m_sim_simple + b_sim_simple + dex_sim_simple, c='r', ls='--', lw=1)
    ax2[1].plot(Lplot, Lplot * m_sim_simple + b_sim_simple - dex_sim_simple, c='r', ls='--', lw=1)


    for i in range(noreals):
        j = np.random.randint(len(samples_sim_simple['b']))

        b, m, dex= samples_ind_simple['b'][j], samples_ind_simple['m'][j], samples_ind_simple['dex'][j]
        #ax2[0].plot(Lplot, Lplot * m + b, c='r', ls='-', lw=1, alpha=0.01)
        ax2[0].fill_between(Lplot, b + Lplot*m + dex, b + Lplot*m - dex, color="r", alpha=0.004*k, zorder = -10)

        b, m, dex= samples_sim_simple['b'][j], samples_sim_simple['m'][j], samples_sim_simple['dex'][j]
        #ax2[1].plot(Lplot, Lplot * m + b, c='r', ls='-', lw=1, alpha=0.01)
        ax2[1].fill_between(Lplot, b + Lplot * m + dex, b + Lplot*m - dex, color="r", alpha=0.004*k, zorder = -10)

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

    ax2[0].plot(Lplot, Lplot * m_ind_mix + b_ind_mix, c='b', ls='-', lw=1,
               label='Mixture Model (Median Values)')
    ax2[0].plot(Lplot, Lplot * m_ind_mix + b_ind_mix + dex_ind_mix, c='b', ls='--', lw=1)
    ax2[0].plot(Lplot, Lplot * m_ind_mix + b_ind_mix - dex_ind_mix, c='b', ls='--', lw=1)

    ax2[1].plot(Lplot, Lplot * m_sim_mix + b_sim_mix, c='b', ls='-', lw=1,
               label='Mixture Model (Median Values)')
    ax2[1].plot(Lplot, Lplot * m_sim_mix + b_sim_mix + dex_sim_mix, c='b', ls='--', lw=1)
    ax2[1].plot(Lplot, Lplot * m_sim_mix + b_sim_mix - dex_sim_mix, c='b', ls='--', lw=1)

    #============

fig2.savefig(fname="./RL-"+CASE+"with-regs.png", format='png')

#---------------------------------------------------------
print("Done!")
plt.show()