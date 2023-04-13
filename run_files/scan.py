'''
scan.py

Example of a gridsearch procedure

HM 9/4
'''

#============================================
import sys

import data_utils

sys.path.append("..")
import numpy as np

from eval_and_opt import loss
from data_utils import lc_to_banded, flatten_dict, array_to_lc
import jax
import jax.numpy as jnp

import matplotlib.pylab as plt
from copy import deepcopy as copy
import config
#============================================

if __name__=="__main__":
    print("Starting unit tests")

    # load some example data
    rootfol = "../Data/data_fake/150day-bad/"
    rootfol = "../Data/real_data/19-A-2925858108/"

    truelag1=150
    truelag2=150

    cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
    line1 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))
    line2 = array_to_lc(np.loadtxt(rootfol + "Hbeta.dat"))

    #Make into banded format
    data  = lc_to_banded([cont, line1,line2])
    data = data_utils.data_tform(data, data_utils.normalize_tform(data))

    fixed_params = data_utils.default_params(np.max(data["bands"]))
    '''
    fixed_params["log_tau"] = 2.9
    fixed_params["log_sigma_c"] = 0.55
    fixed_params["rel_amps"] = jnp.array([0.9,0.4])
    fixed_params["means"] = jnp.array([-2.3,0.55,0])
    '''


    gridloss = lambda x,y: loss(data, fixed_params | {"lags": jnp.array([x,y])})
    #gridloss = lambda x,y: loss(data, {"lags": jnp.array([x,160]), "log_tau": y})

    nplot = 256
    lag1 = np.linspace(0,config.lag_max, nplot)
    lag2 = np.linspace(0,config.lag_max, nplot)
    #log_taus = np.linspace(config.log_tau_min, config.log_tau_max, nplot)

    X = lag1
    Y = lag2

    true_x = 540
    true_y = 540

    xlabel = "$\Delta t_{1}$"
    ylabel = "$\Delta t_{2}$"
    #ylabel = "$\ ln| \\tau |$"

    Z = np.zeros([nplot,nplot])
    for i in range(nplot):
        if i%50==0: print(i)
        for j in range(nplot):
            x= X[i]
            y= Y[j]

            Z[i,j] = gridloss(x,y)
    Z=Z.T
    Z=Z[::-1,:]

    #-----------------------------------
    plt.figure(figsize=(4,4))
    plt.imshow(Z,
               interpolation='none', cmap='viridis',
               extent=[min(X),max(X),min(Y),max(Y)],
               aspect="auto")
    plt.axvline(true_x)
    plt.axhline(true_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Log Probability")
    plt.tight_layout()
    plt.show()

    Z2 =Z - np.max(Z)
    Z3 = copy(Z2)

    Z2 = np.clip(Z2,a_min=-30, a_max = 0)

    #-----------------------------------
    plt.figure(figsize=(4,4))
    plt.imshow(Z2,
               interpolation='none', cmap='viridis',
               extent=[min(X),max(X),min(Y),max(Y)],
               aspect="auto")
    plt.axvline(true_x)
    plt.axhline(true_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Log Probability, Filtered")
    plt.tight_layout()
    plt.show()

    #-----------------------------------
    contrast = 1
    Z3 = np.exp(Z3*contrast)

    plt.figure(figsize=(4,4))
    plt.imshow(Z3,
               interpolation='none', cmap='viridis',
               extent=[min(X),max(X),min(Y),max(Y)],
               aspect="auto")
    plt.axvline(true_x)
    plt.axhline(true_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Likelihood")
    plt.tight_layout()
    plt.show()

    #-----------------------------------
    fig,ax=plt.subplots(2,1)
    ax[0].plot(X, np.sum(Z3, axis=0)* (X[1]-X[0]))
    ax[1].plot(Y, np.sum(Z3, axis=1)* (Y[1]-Y[0]))

    norm = loss(data, fixed_params | {"lags": jnp.array([10000,10000])})
    norm = np.exp(norm)
    print(norm)
    plt.show()