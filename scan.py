
from argparse import ArgumentParser
import numpy as np

from eval_and_opt import loss
from data_utils import lc_to_banded, flatten_dict, array_to_lc
import jax
import jax.numpy as jnp

import matplotlib.pylab as plt
from copy import deepcopy as copy

if __name__=="__main__":
    print("Starting unit tests")

    # load some example data
    rootfol = "./Data/data_fake/180day-bad/"

    truelag1=180
    truelag2=180



    cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
    line1 = array_to_lc(np.loadtxt(rootfol + "line1.dat"))
    line2 = array_to_lc(np.loadtxt(rootfol + "line2.dat"))

    #Make into banded format
    data  = lc_to_banded([cont, line1, line2])

    gridloss = lambda x,y: loss(data, {"lags": jnp.array([x,y])})

    nplot = 128
    lag1 = np.linspace(0,500, nplot)
    lag2 = np.linspace(0,500, nplot)

    Z = np.zeros([nplot,nplot])
    for i in range(nplot):
        if i%50==0: print(i)
        for j in range(nplot):
            x= lag1[i]
            y= lag2[j]

            Z[i,j] = gridloss(x,y)
    Z=Z.T
    Z=Z[::-1,:]

    plt.imshow(Z,
               interpolation='none', cmap='viridis',
               extent=[min(lag1),max(lag1),min(lag1),max(lag1)])
    plt.axvline(truelag1)
    plt.axhline(truelag2)
    plt.xlabel("$\Delta t_{1}$")
    plt.ylabel("$\Delta t_{2}$")
    plt.title("Log Probability")
    plt.tight_layout()
    plt.show()

    Z2=Z - np.max(Z)
    Z2=np.clip(Z2, a_min=-30,a_max = 0)

    plt.figure()
    plt.imshow(Z2,
               interpolation='none', cmap='viridis',
               extent=[min(lag1),max(lag1),min(lag1),max(lag1)])
    plt.axvline(truelag1)
    plt.axhline(truelag2)
    plt.xlabel("$\Delta t_{1}$")
    plt.ylabel("$\Delta t_{2}$")
    plt.title("Log Probability, Filtered")
    plt.tight_layout()
    plt.show()