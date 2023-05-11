'''
scan.py

Example of a gridsearch procedure

HM 9/4
'''

#============================================
import sys

sys.path.append("C:/Users/hughm/My Drive/HonoursThesis/HM_Lag_Recovery")
sys.path.append("..")

import data_utils

import numpy as np

from eval_and_opt import loss, realize
from data_utils import lc_to_banded, flatten_dict, array_to_lc
import jax
import jax.numpy as jnp

import matplotlib.pylab as plt
from copy import deepcopy as copy
import config
from scipy.integrate import simpson
#============================================

if __name__=="__main__":
    print("Beginning signal gen")

    # load some example data
    targ = "79-A-2970604169"
    rootfol = "../Data/real_data/"
    rootfol+= targ
    rootfol+="/"

    # load some example data
    targ = "81-24-08"
    rootfol = "../Data/data_shuffle/"
    rootfol+= targ
    rootfol+="/"

    try:
        cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
        line1 = array_to_lc(np.loadtxt(rootfol + "Hbeta.dat"))
        line2 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))

        #Make into banded format
        data  = lc_to_banded([cont, line1,line2])
        data = data_utils.data_tform(data, data_utils.normalize_tform(data))
    except:
        try:
            cont = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
            line1 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))
            line2 = array_to_lc(np.loadtxt(rootfol + "CIV.dat"))

            # Make into banded format
            data = lc_to_banded([cont, line1, line2])
            data = data_utils.data_tform(data, data_utils.normalize_tform(data))
        except:
            cont = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
            line1 = array_to_lc(np.loadtxt(rootfol + "line1.dat"))
            line2 = array_to_lc(np.loadtxt(rootfol + "line2.dat"))

            # Make into banded format
            data = lc_to_banded([cont, line1, line2])
            data = data_utils.data_tform(data, data_utils.normalize_tform(data))




    #========
    try:
        DATA = np.loadtxt(rootfol+"outchain.dat")
        NEST_SEED = np.loadtxt(rootfol+"outchain-nest-seed.dat")
        NEST_FULL = np.loadtxt(rootfol+"outchain-nest-full.dat")
        KEYS = np.loadtxt(rootfol+"outchain_keys.dat", dtype="str_")
    except:
        DATA = np.loadtxt(rootfol+"outchain-twoline.dat")
        NEST_SEED = np.loadtxt(rootfol+"outchain-nest-seed-twoline.dat")
        NEST_FULL = np.loadtxt(rootfol+"outchain-nest-full-twoline.dat")
        KEYS = np.loadtxt(rootfol+"outchain_keys-twoline.dat", dtype="str_")

    #------------------------------------------------------------------------

    n_samples = 64
    n_realspersample = 1

    n_grid = 256

    baseline = np.max(data['T']) - np.min(data['T'])
    Tplot = np.linspace(-baseline*0.25, baseline*1.25, n_grid)
    fig, ax = plt.subplots(3,1, figsize=(10,5), sharex=True)

    colours = ['b','r','g']

    for band_no in range(3):

        bandinds = np.where(data['bands'] == band_no)[0]
        ax[band_no].errorbar(data['T'][bandinds], data['Y'][bandinds], yerr=data['E'][bandinds],
                     fmt='none',
                     capsize=2,
                     c=colours[band_no]
                             )

        print("Doing plots for band %i" % band_no)
        for i in range(n_samples):
            print("\t Doing plot: ", i)
            sample_no = np.random.randint(DATA.shape[0])

            params = {'lags': jnp.array([DATA[sample_no,0], DATA[sample_no,1]]),
                      'log_sigma_c': DATA[sample_no,2],
                      'log_tau': DATA[sample_no, 3],
                      'means': jnp.array([DATA[sample_no, 4], DATA[sample_no, 5], DATA[sample_no, 6]]),
                      'rel_amps': jnp.array([DATA[sample_no, 7], DATA[sample_no, 8]]),
                      }


            Yplot = realize(data, Tout =Tplot, params=params, band=band_no, seed = i, nreals=n_realspersample)

            lagshift = 0
            if band_no!=0: lagshift=params['lags'][band_no-1]

            if n_realspersample>1:
                for j in range(n_realspersample):
                    ax[band_no].plot(Tplot+lagshift, Yplot[j,:], lw=0.1, c='k', alpha=0.5, zorder = -10)
            else:
                ax[band_no].plot(Tplot+lagshift, Yplot, lw=0.1, c='k', alpha=0.5, zorder = -10)

    fig.suptitle("MCMC Constrained Realizations for %s" %targ)
    fig.supxlabel("Time (days)")
    fig.supylabel("Signal Variation")
    ax[0].set_xlim(0, baseline * 1.1)

    line1name, line2name = "Line 1", "Line 2"
    if targ[3] == 'A': line1name, line2name = '$H \\beta $', 'MGII'
    if targ[3] == 'B': line1name, line2name = 'MgII', 'CIV'
    ax[0].set_title("Continuum")
    ax[1].set_title(line1name)
    ax[2].set_title(line2name)

    fig.tight_layout()

    plt.show()