
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pylab as plt

SIGNAL = np.loadtxt("banded_data.dat")

#==================================

DATA = np.loadtxt("outchain.dat")
NEST_SEED = np.loadtxt("outchain-nest-seed.dat")
NEST_FULL = np.loadtxt("outchain-nest-full.dat")
KEYS = np.loadtxt("outchain_keys.dat", dtype="str_")

c = ChainConsumer()
c.add_chain(DATA, parameters = list(KEYS), name = "HMC")
c.add_chain(NEST_SEED, parameters = list(KEYS), name = "Nest Seeds")
c.add_chain(NEST_FULL, parameters = list(KEYS), name = "Nest Full")

truth = {"rel_amps_0" : 1,
         "rel_amps_1" : 1,
         "log_sigma_c" : 0,
         "log_tau" : 5.99,
         "means_0":0,
         "means_1":0,
         "means_2":0}

extents = {"amps_0" :       (0,10),
         "amps_1" :         (0,10),
         "log_sigma_c" :    (-2.5,2.5),
         "log_tau" :        (3,15),
         "means_0":         (-20,20),
         "means_1":         (-20,20),
         "means_2":         (-20,20),
         "lags_1":          (0,800),
         "lags_2":          (0,800)
           }

c.plotter.plot(filename ="./contours.png", truth=truth, extents=extents)

#==================================
fig,ax = plt.subplots(2,1)
cols = ['b','r','g']

for file,c in zip(["cont.dat", "line1.dat", "line2.dat"],cols):
    data = np.loadtxt(file)
    ax[0].errorbar(data[:,0],   data[:,1],  yerr=data[:,2], fmt='none', c=c)
    
    
ax[0].set_ylim([-5,5])
ax[0].axhline(0,c='k', ls='--', lw=0.5)

for band,c in zip([0,1,2], cols):
    inds = np.where(SIGNAL[:,3]==band)[0]
    
    ax[1].errorbar(SIGNAL[:,0][inds],SIGNAL[:,1][inds],yerr=SIGNAL[:,2][inds],fmt='none',c=c)
    
    
ax[1].set_ylim([-5,5])
ax[1].axhline(0,c='k', ls='--', lw=0.5)

fig.savefig("signal.png",fmt="png")
