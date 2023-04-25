import sys
sys.path.append("..")

import os
import SIMBA
os.chdir('..')

import numpy as np
import matplotlib.pylab as plt

#-------------------------

table_url = "SIMBA_jobstatus.dat"
njobs = 93
nbins = 32
COVARS = np.zeros([njobs,3])

#-------------------------
for i in range(93):
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

#-------------------------
os.chdir('./processing')
np.savetxt('line_consistency_check.dat', COVARS, fmt=['%i','%.3f','%.3f'], delimiter='\t')

