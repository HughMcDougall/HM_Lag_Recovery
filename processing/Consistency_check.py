import sys
sys.path.append("..")

import os
import SIMBA
os.chdir('..')

import numpy as np
import matplotlib.pylab as plt

from math import exp

#-------------------------

table_url = "SIMBA_jobstatus.dat"
njobs = 93
nbins = 32

do_correl = True
do_lags = True

COVARS = np.zeros([njobs,3])
LAGS = np.zeros([njobs,5])

#-------------------------
for i in range(njobs):
    args = SIMBA.get_args(i, table_url)
    targ_url = args["out_url"]

    print(i, targ_url)

    #Load in lags
    try:
        DATA_BOTH  = np.loadtxt(targ_url+"outchain.dat")[:,:2]
        DATA_LINE1 = np.loadtxt(targ_url+"outchain-line1.dat")[:,0]
        DATA_LINE2 = np.loadtxt(targ_url+"outchain-line2.dat")[:,0]
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
        lag_1_var   = np.percentile(DATA_LINE1, 100*(1-exp(-1)/2))-np.percentile(DATA_LINE1, 100*exp(-1)/2)

        lag_2       = np.median(DATA_LINE2)
        lag_2_var   = np.percentile(DATA_LINE2, 100*(1-exp(-1)/2))-np.percentile(DATA_LINE2, 100*exp(-1)/2)

        LAGS[i,0] = i
        LAGS[i,1] = lag_1
        LAGS[i,2] = lag_1_var
        LAGS[i,3] = lag_2
        LAGS[i,4] = lag_2_var
#-------------------------
os.chdir('./processing')
if do_correl: np.savetxt('line_consistency_check.dat', COVARS, fmt=['%i','%.3f','%.3f'], delimiter='\t')
if do_lags: np.savetxt('line_recovered_lags.dat', LAGS, fmt=['%i','%.3f','%.3f','%.3f','%.3f'], delimiter='\t')
