#-------------------------

#-------------------------
import sys
sys.path.append("..")

import os
import SIMBA
os.chdir('..')

import numpy as np
import matplotlib.pylab as plt

from math import exp

#-------------------------
def getpeak(data):
    h = np.histogram(data, bins = 64)
    hData = h[0]
    i = np.where(hData == np.max(hData))
    return(np.mean(h[1][i]))

#-------------------------

table_url = "SIMBA_jobstatus.dat"
njobs = 93
nbins = 32

do_correl   = False
do_lags     = False
do_taus     = True

COVARS = np.zeros([njobs,3])
LAGS   = np.zeros([njobs,13])
TAUS   = np.zeros([njobs,7])

#-------------------------
for i in range(njobs):
    args = SIMBA.get_args(i, table_url)
    targ_url = args["out_url"]

    print(i, targ_url)

    #Load in lags
    if do_correl or do_lags:
        try:
            DATA_BOTH  = np.loadtxt(targ_url+"outchain.dat")[:,:2]
            DATA_LINE1 = np.loadtxt(targ_url+"outchain-line1.dat")[:,0]
            DATA_LINE2 = np.loadtxt(targ_url+"outchain-line2.dat")[:,0]
        except:
            continue
    if do_taus:
        try:
            TAU_BOTH = np.loadtxt(targ_url+"outchain.dat")[:,3]
            TAU_LINE1= np.loadtxt(targ_url+"outchain-line1.dat")[:,2]
            TAU_LINE2= np.loadtxt(targ_url+"outchain-line2.dat")[:,2]
        except:
            continue

    if do_correl:
        #Make histograms
        lag1_hist_ind = np.histogram(DATA_LINE1, bins = nbins, range = [0,800], density=True)[0]
        lag2_hist_ind = np.histogram(DATA_LINE2, bins = nbins, range = [0,800], density=True)[0]

        lag1_hist_sim = np.histogram(DATA_BOTH[:,0], bins = nbins, range = [0,800], density=True)[0]
        lag2_hist_sim = np.histogram(DATA_BOTH[:,1], bins = nbins, range = [0,800], density=True)[0]

        #Calculate covariances
        covar_lag1  = np.sum(lag1_hist_ind*lag1_hist_sim) / np.sqrt(np.sum(lag1_hist_ind*lag1_hist_ind) * np.sum(lag1_hist_sim*lag1_hist_sim))
        covar_lag2  = np.sum(lag2_hist_ind*lag2_hist_sim) / np.sqrt(np.sum(lag2_hist_ind*lag2_hist_ind) * np.sum(lag2_hist_sim*lag2_hist_sim))
        
        #Save to covar vector
        COVARS[i,0]=i
        COVARS[i,1]=covar_lag1
        COVARS[i,2]=covar_lag2

    if do_lags:
        #Recover Lags
        lag_1       = np.median(DATA_LINE1)
        lag_1_var   = np.percentile(DATA_LINE1, 84.13)-np.percentile(DATA_LINE1, 15.87)
        lag_1_peak  = getpeak(DATA_LINE1)

        lag_2       = np.median(DATA_LINE2)
        lag_2_var   = np.percentile(DATA_LINE2, 84.13)-np.percentile(DATA_LINE2, 15.87)
        lag_2_peak  = getpeak(DATA_LINE2)

        simlag_1       = np.median(DATA_BOTH[:,0])
        simlag_1_var   = np.percentile(DATA_BOTH[:,0], 84.13)-np.percentile(DATA_BOTH[:,0], 15.87)
        simlag_1_peak  = getpeak(DATA_BOTH[:,0])

        simlag_2       = np.median(DATA_BOTH[:,1])
        simlag_2_var   = np.percentile(DATA_BOTH[:,1], 84.13)-np.percentile(DATA_BOTH[:,1], 15.87)
        simlag_2_peak  = getpeak(DATA_BOTH[:,1])

        LAGS[i,0] = i

        LAGS[i,1] = lag_1_peak
        LAGS[i,2] = lag_1s
        LAGS[i,3] = lag_1_var

        LAGS[i,4] = lag_2_peak
        LAGS[i,5] = lag_2
        LAGS[i,6] = lag_2_var

        LAGS[i,7] = simlag_1_peak
        LAGS[i,8] = simlag_1
        LAGS[i,9] = simlag_1_var

        LAGS[i,10] = simlag_2_peak
        LAGS[i,11] = simlag_2
        LAGS[i,12] = simlag_2_var

    if do_taus:

        TAU_LINES = np.concatenate([TAU_LINE1,TAU_LINE2])

        TAUS[i,0] = i

        TAUS[i,1] = getpeak(TAU_BOTH)
        TAUS[i,2] = np.median(TAU_BOTH)
        TAUS[i,3] = np.percentile(TAU_BOTH, 84.13)-np.percentile(TAU_BOTH, 15.87)

        TAUS[i,4] = getpeak(TAU_LINES)
        TAUS[i,5] = np.median(TAU_LINES)
        TAUS[i,6] = np.percentile(TAU_LINES, 84.13)-np.percentile(TAU_LINES, 15.87)
            


#-------------------------
print("Saving outputs")
os.chdir('./processing')
if do_correl: np.savetxt('line_consistency_check.dat', COVARS, fmt=['%i','%.3f','%.3f'], delimiter='\t')
if do_lags: np.savetxt('line_recovered_lags.dat', LAGS, fmt=['%i'] + ['%.3f'] * (LAGS.shape[-1]-1), delimiter='\t')
if do_taus: np.savetxt('line_contscales.dat', TAUS, fmt=['%i'] + ['%.3f'] * (TAUS.shape[-1]-1), delimiter='\t')
print("All done.")
