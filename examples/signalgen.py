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

    targ = "92-B-2971214955"
    rootfol = "../Data/real_data/"
    rootfol+= targ
    rootfol+="/"

    print("target = %s" %targ)

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

    Y_cont  = np.zeros([n_samples*n_realspersample,n_grid])
    Y_line1 = np.zeros([n_samples*n_realspersample,n_grid])
    Y_line2 = np.zeros([n_samples*n_realspersample,n_grid])

    baseline = np.max(data['T']) - np.min(data['T'])
    Tplot = np.linspace(-baseline*0.25, baseline*1.25, n_grid)

    for band_no in range(3):

        print("Doing realizations for band %i" % band_no)
        for i in range(n_samples):
            print("\t Doing Realization: ", i)
            sample_no = np.random.randint(DATA.shape[0])

            params = {'lags': jnp.array([DATA[sample_no,0], DATA[sample_no,1]]),
                      'log_sigma_c': DATA[sample_no,2],
                      'log_tau': DATA[sample_no, 3],
                      'means': jnp.array([DATA[sample_no, 4], DATA[sample_no, 5], DATA[sample_no, 6]]),
                      'rel_amps': jnp.array([DATA[sample_no, 7], DATA[sample_no, 8]]),
                      }

            Yplot = realize(data, Tout =Tplot, params=params, band=band_no, seed = i, nreals=n_realspersample)

            if band_no==0:
                Y_cont[n_realspersample * i : n_realspersample * (i+1),:] = Yplot
            elif band_no==1:
                Y_line1[n_realspersample * i : n_realspersample * (i+1),:] = Yplot
            elif band_no==2:
                Y_line2[n_realspersample * i : n_realspersample * (i+1),:] = Yplot

    #MASKING
    Y_cont = np.ma.array(Y_cont, mask=np.isnan(Y_cont))
    Y_line1 = np.ma.array(Y_line1, mask=np.isnan(Y_line1))
    Y_line2 = np.ma.array(Y_line2, mask=np.isnan(Y_line2))

    #PLOTTING
    colours = ['navy', 'maroon', 'darkgreen']
    lw = 5.0
    lwk = 2.0
    alpha = 0.1
    alphak = 0.025 / lwk
    c2 = ['dodgerblue', 'coral', 'chartreuse']

    fig, ax = plt.subplots(3,1, figsize=(8,4), sharex=True, sharey=False)
    for band_no in range(3):

        bandinds = np.where(data['bands'] == band_no)[0]
        ax[band_no].errorbar(data['T'][bandinds], data['Y'][bandinds], yerr=data['E'][bandinds],
                     fmt='none',
                     capsize=2,
                     c=colours[band_no]
                             )

        print("Doing plots for band %i" % band_no)
        for i in range(n_samples*n_realspersample):
            print("\t Doing plot: ", i)

            lagshift = 0
            if band_no!=0: lagshift=params['lags'][band_no-1]

            for Y,j in zip([Y_cont,Y_line1,Y_line2],range(3)):
                if band_no!=j:
                    continue
                else:
                    ax[band_no].plot(Tplot+lagshift, Y[i,:], lw=lw, c=c2[j], alpha=alpha, zorder = -10)
                    ax[band_no].plot(Tplot+lagshift, Y[i,:], lw=lwk, c='k', alpha=alphak, zorder = -5)

    fig.suptitle("MCMC Constrained Realizations for %s" %targ)
    fig.supxlabel("Time (days)")
    fig.supylabel("Signal Variation")
    ax[0].set_xlim(0, baseline * 1.1)

    scale = 3.0
    ax[0].set_ylim(np.mean(Y_cont)-np.std(Y_cont)*scale,np.mean(Y_cont)+np.std(Y_cont)*scale)
    ax[1].set_ylim(np.mean(Y_line1)-np.std(Y_line1)*scale,np.mean(Y_line1)+np.std(Y_line1)*scale)
    ax[2].set_ylim(np.mean(Y_line2)-np.std(Y_line2)*scale,np.mean(Y_line2)+np.std(Y_line2)*scale)

    line1name, line2name = "Line 1", "Line 2"
    if targ[3] == 'A': line1name, line2name = '$H \\beta $', 'MgII'
    if targ[3] == 'B': line1name, line2name = 'MgII', 'CIV'
    ax[0].set_title("Continuum")
    ax[1].set_title(line1name)
    ax[2].set_title(line2name)

    fig.tight_layout()

    plt.savefig(fname = "signal-"+targ.replace('/','-')+".png")
    plt.show()