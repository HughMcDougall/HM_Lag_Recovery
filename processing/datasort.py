'''
Collates chains with low variance
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import sys
import pandas as pd
from copy import deepcopy as copy
from jax.random import PRNGKey

import numpyro

import matplotlib.pylab as plt

sys.path.append("..")

#------------------------------------

#------------------------------------
SUMMARY = pd.read_csv("./summary.dat", sep='\t', index_col=0)

TARG_FOLS   = np.array(SUMMARY["Folder"])
Z           = np.array(SUMMARY["redshift"])
LOGLUM      = np.array(SUMMARY["L2"])
LINETYPE    = np.array(SUMMARY["Type"])

rootfol="../"
N = len(TARG_FOLS)
#------------------------------------
#Get recovered lag samples

lag_1_samples_sim = [None]*N
lag_2_samples_sim = [None]*N
lag_1_samples_ind = [None]*N
lag_2_samples_ind = [None]*N

HBETA_samples_sim = []
MGII_samples_sim = []
CIV_samples_sim = []

HBETA_samples_ind = []
MGII_samples_ind = []
CIV_samples_ind = []

HBETA_lums = []
MGII_lums = []
CIV_lums = []

HBETA_zs = []
MGII_zs = []
CIV_zs = []

sparseness = 100

print("Doing data load")
for i in range(N):
    line_type = LINETYPE[i]
    print("\t %i: \t %s" %(i, line_type))

    data_sim  = np.loadtxt(rootfol+TARG_FOLS[i]+"outchain.dat")
    data_ind1 = np.loadtxt(rootfol+TARG_FOLS[i]+"outchain-line1.dat")
    data_ind2 = np.loadtxt(rootfol+TARG_FOLS[i]+"outchain-line2.dat")

    Y1_sim = data_sim[:,0][::sparseness*2]
    Y2_sim = data_sim[:,1][::sparseness*2]
    Y1_ind = data_ind1[:,0][::sparseness]
    Y2_ind = data_ind2[:,0][::sparseness]



    if line_type == "Hbeta/MGII":
        HBETA_lums.append(LOGLUM[i])
        MGII_lums.append(LOGLUM[i])

        HBETA_zs.append(Z[i])
        MGII_zs.append(Z[i])

        HBETA_samples_sim.append(Y1_sim)
        HBETA_samples_ind.append(Y1_ind)

        MGII_samples_sim.append(Y2_sim)
        MGII_samples_ind.append(Y2_ind)


    elif line_type == "MGII/CIV":
        MGII_lums.append(LOGLUM[i])
        CIV_lums.append(LOGLUM[i])

        MGII_zs.append(Z[i])
        CIV_zs.append(Z[i])

        MGII_samples_sim.append(Y1_sim)
        MGII_samples_ind.append(Y1_ind)

        CIV_samples_sim.append(Y2_sim)
        CIV_samples_ind.append(Y2_ind)
    else:
        print("Type error on entry %i" %i)

print("Done.")

#--------------------------------------------
np.savetxt(X=np.array(HBETA_samples_sim).T, fname = "./results/HBETA_samples_sim.dat", delimiter="\t")
np.savetxt(X=np.array(MGII_samples_sim).T, fname = "./results/MGII_samples_sim.dat", delimiter="\t")
np.savetxt(X=np.array(CIV_samples_sim).T, fname = "./results/CIV_samples_sim.dat", delimiter="\t")

np.savetxt(X=np.array(HBETA_samples_ind).T, fname = "./results/HBETA_samples_ind.dat", delimiter="\t")
np.savetxt(X=np.array(MGII_samples_ind).T, fname = "./results/MGII_samples_ind.dat", delimiter="\t")
np.savetxt(X=np.array(CIV_samples_ind).T, fname = "./results/CIV_samples_ind.dat", delimiter="\t")

np.savetxt(X=HBETA_lums, fname = "./results/HBETA_lums.dat", delimiter="\t")
np.savetxt(X=MGII_lums, fname = "./results/MGII_lums.dat", delimiter="\t")
np.savetxt(X=CIV_lums, fname = "./results/CIV_lums.dat", delimiter="\t")

np.savetxt(X=HBETA_zs, fname = "./results/HBETA_zs.dat", delimiter="\t")
np.savetxt(X=MGII_zs, fname = "./results/MGII_zs.dat", delimiter="\t")
np.savetxt(X=CIV_zs, fname = "./results/CIV_zs.dat", delimiter="\t")
