'''
scan.py

Example of a gridsearch procedure

HM 9/4
'''

#============================================
import sys

sys.path.append("C:/Users/hughm/My Drive/HonoursThesis/HM_Lag_Recovery")
sys.path.append("....")
import data_utils

import numpy as np

from eval_and_opt import loss
from data_utils import lc_to_banded, flatten_dict, array_to_lc
import jax
import jax.numpy as jnp

import matplotlib.pylab as plt
from copy import deepcopy as copy
import config
from scipy.integrate import simpson
#============================================

if __name__=="__main__":
    print("Beginning scan")

    # load some example data
    rootfol = "./"

    cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
    line1 = array_to_lc(np.loadtxt(rootfol + "Hbeta.dat"))
    line2 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))

    #Make into banded format
    data  = lc_to_banded([cont, line1,line2])
    data = data_utils.data_tform(data, data_utils.normalize_tform(data))

    fixed_params = data_utils.default_params(np.max(data["bands"]))
    fixed_params = fixed_params | {
        "rel_amps":jnp.array([1.16,4.7]),
        "means": jnp.array([0.28,0.12,-0.0])
    }

    #gridloss = lambda t1,t2,ltau: loss(data, fixed_params | {"lags": jnp.array([t1,t2])})
    gridloss = lambda t1,t2,ltau,ramp2: loss(data, fixed_params | {"lags": jnp.array([t1,t2]), "log_sigma_c":0.05 + (ltau-4.91)*0.5, "log_tau": ltau, "rel_amps":jnp.array([1.16, ramp2])})


    nplot = 128
    nint  = 20
    lag1 = np.linspace(0,config.lag_max, nplot)
    lag2 = np.linspace(0,config.lag_max, nplot)
    #log_taus = np.linspace(4, 7.5, nint)
    log_taus = np.linspace(4, 7.5, nint)
    ramp2s   = np.linspace(2, 8, nint)

    X = lag1
    Y = lag2

    true_x = 350
    true_y = 360

    xlabel = "$\Delta t_{1}$"
    ylabel = "$\Delta t_{2}$"

    Z_full = np.zeros([nplot,nplot,nint,nint])
    for i in range(nplot):
        if i%1==0: print("%i / %i" %(i,nplot))
        for j in range(nplot):

            for k in range(nint):
                for l in range(nint):
                    x = X[i]
                    y = Y[j]
                    z1 = log_taus[k]
                    z2 = ramp2s[l]

                    Z_full[i,j,k,l]=gridloss(x,y,z1,z2)
                
    # Convert LL to exp(LL) and integrate for depth,
    Z_full = Z_full - np.max(Z_full)

    Z = np.exp(Z_full)
    if Z.shape[-1]>1:
        Z = simpson(Z, axis=2, dx = (log_taus[1]-log_taus[0]))
        Z = simpson(Z, axis=2, dx=(ramp2s[1] - ramp2s[0]))
    else:
        Z = Z[:,:,0,0]
    # Then switch back for consistency with old plotting methods

    Z = np.log(Z)
    Z = Z
    Z = Z[::-1,:]

    #-----------------------------------
    plt.figure(figsize=(4,4))
    plt.imshow(Z,
               interpolation='none', cmap='viridis',
               extent=[min(X),max(X),min(Y),max(Y)],
               aspect="auto")
    plt.title("Log Probability")

    plt.tight_layout()
    plt.savefig("./LogProb.png", format='png')
    plt.show()



    #-----------------------------------
    Z2 = Z - np.max(Z)
    Z2 = np.clip(Z2,a_min=-300, a_max = 0)
    plt.figure(figsize=(4,4))
    plt.imshow(Z2,
               interpolation='none', cmap='viridis',
               extent=[min(X),max(X),min(Y),max(Y)],
               aspect="auto")
    plt.title("Log Probability, Filtered")

    plt.tight_layout()
    plt.savefig("./LogProb_filtered.png", format='png')
    plt.show()

    #-----------------------------------
    Z3 = Z - np.max(Z)
    contrast = 0.1
    Z3 = np.exp(Z3*contrast)

    plt.figure(figsize=(4,4))
    plt.imshow(Z3,
               interpolation='none', cmap='viridis',
               extent=[min(X),max(X),min(Y),max(Y)],
               aspect="auto")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Likelihood")

    plt.tight_layout()
    plt.savefig("./Prob.png", format='png')
    plt.show()

    #-----------------------------------
    fig,ax=plt.subplots(2,1, figsize=(4,3), sharex=True, sharey=True)
    fig.suptitle("Marginalized Likelihoods")
    int1 = simpson(Z3, axis=1)* (Y[1]-Y[0])
    int2 = simpson(Z3, axis=0)* (X[1]-X[0])

    int1/=simpson(int1,dx = (X[1]-X[0]))
    int2/=simpson(int2,dx = (Y[1]-Y[0]))

    ax[0].plot(X, int2, c = 'k')
    ax[1].plot(Y[::-1], int1, c = 'k')

    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(ylabel)

    ax[0].set_xlim(min(X),max(X))
    ax[1].set_xlim(min(Y),max(Y))


    plt.tight_layout()
    plt.savefig("./Prob_Marginal.png", format='png')
    plt.show()
