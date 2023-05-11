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
    rootfol = "./"

    try:
        cont  = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
        line1 = array_to_lc(np.loadtxt(rootfol + "Hbeta.dat"))
        line2 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))

        #Make into banded format
        data  = lc_to_banded([cont, line1,line2])
        data = data_utils.data_tform(data, data_utils.normalize_tform(data))
    except:
        cont = array_to_lc(np.loadtxt(rootfol + "cont.dat"))
        line1 = array_to_lc(np.loadtxt(rootfol + "MGII.dat"))
        line2 = array_to_lc(np.loadtxt(rootfol + "CIV.dat"))

        # Make into banded format
        data = lc_to_banded([cont, line1, line2])
        data = data_utils.data_tform(data, data_utils.normalize_tform(data))


    #========
    try:
        DATA = np.loadtxt("outchain.dat")
        NEST_SEED = np.loadtxt("outchain-nest-seed.dat")
        NEST_FULL = np.loadtxt("outchain-nest-full.dat")
        KEYS = np.loadtxt("outchain_keys.dat", dtype="str_")
    except:
        DATA = np.loadtxt("outchain-twoline.dat")
        NEST_SEED = np.loadtxt("outchain-twoline-nest-seed.dat")
        NEST_FULL = np.loadtxt("outchain-twoline-nest-full.dat")
        KEYS = np.loadtxt("outchain_keys-twoline.dat", dtype="str_")

    #------------------------------------------------------------------------

    n_samples = 16
    n_realspersample = 4
    band = 0

    n_grid = 256

    Tplot = np.linspace(np.min(data['T']), np.max(data['T']), n_grid)

    plt.figure()
    bandinds = np.where(data['bands']==band)[0]
    plt.errorbar(data['T'][bandinds],data['Y'][bandinds],yerr=data['E'][bandinds],fmt='none',capsize=2,label='data')
    for i in range(n_samples):
        print("\t Doing plot: ", i)
        sample_no = np.random.randint(DATA.shape[0])

        params = {'lags': jnp.array([DATA[sample_no,0], DATA[sample_no,1]]),
                  'log_sigma_c': DATA[sample_no,2],
                  'log_tau': DATA[sample_no, 3],
                  'means': jnp.array([DATA[sample_no, 4], DATA[sample_no, 5], DATA[sample_no, 6]]),
                  'rel_amps': jnp.array([DATA[sample_no, 7], DATA[sample_no, 8]]),
                  }

        Yplot = realize(data, Tout =Tplot, params=params, band=band, seed = i, nreals=n_realspersample)

        if n_realspersample>1:
            for j in range(n_realspersample):
                plt.plot(Tplot,Yplot[j,:], lw=0.1, c='k', zorder = -10)
        else:
            plt.plot(Tplot, Yplot, lw=0.1, c='k', alpha=0.1, zorder = -10)

    plt.show()