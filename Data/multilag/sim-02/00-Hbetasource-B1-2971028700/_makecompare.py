
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pylab as plt

import os

#SIGNAL = np.loadtxt("banded_data.dat")

#==================================

try:
    DATA = np.loadtxt("outchain.dat")
    NEST_SEED = np.loadtxt("outchain-nest-seed.dat")
    NEST_FULL = np.loadtxt("outchain-nest-full.dat")
    KEYS = np.loadtxt("outchain_keys.dat", dtype="str_")
except:
    
    DATA = np.loadtxt("outchain-twoline.dat")
    NEST_SEED = np.loadtxt("outchain-nest-seed-twoline.dat")
    NEST_FULL = np.loadtxt("outchain-nest-full-twoline.dat")
    KEYS = np.loadtxt("outchain_keys-twoline.dat", dtype="str_")


DATA_JAVELIN = np.loadtxt('chain-twoline-highqual.dat')
DATA_JAVELIN = {
    "lags_1": DATA_JAVELIN[:,2],
    "lags_2": DATA_JAVELIN[:,5]
    }

#=======================================
extents = {"amps_0" :       (0,10),
         "amps_1" :         (0,10),
         "log_sigma_c" :    (-2,4),
         "log_tau" :        (2,14),
         "means_0":         (-20,20),
         "means_1":         (-20,20),
         "means_2":         (-20,20),
         "lags_1":          (0,750),
         "lags_2":          (0,750)
           }

truth = {"amps_0" :         1.0,
         "amps_1" :         1.0,
         "log_sigma_c" :    0,
         "log_tau" :        np.log(400),
         "means_0":         0,
         "means_1":         0,
         "means_2":         0,
         "lags_1":          0,
         "lags_2":          0
           }

for sim, lag in zip(['sim-01','sim-02','sim-03','sim-04','sim-05'],[150,250,350,450,550]):
    if sim in os.getcwd():
        truth['lags_1'] = lag
        truth['lags_2'] = lag + 10
        break

xlabel = '$\Delta t_{1}$'
ylabel = '$\Delta t_{2}$'
labels = [xlabel,ylabel]

#=======================================
c = ChainConsumer()
c.add_chain(DATA, parameters = list(KEYS),  name = "LITMUS",    color='blue')
c.add_chain(DATA_JAVELIN,                   name = "JAVELIN",   color='purple')

#=======================================
#COMPARISON

fig,ax = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True)

for axi in ax:
    axi.set_aspect('equal')
    axi.set_aspect('equal')

    axi.set_xlim(extents['lags_1'])
    axi.set_ylim(extents['lags_1'])
    
    axi.axvline(truth['lags_1'], c='k', ls='--', lw=1)
    axi.axhline(truth['lags_2'], c='k', ls='--', lw=1)

fig.supxlabel(xlabel)
fig.supylabel(ylabel)


#c.plotter.plot(filename ="./contours_lagsonly.png", extents=extents, parameters = ["lags_1","lags_2"])
c.plotter.plot_contour(ax[0], "lags_1","lags_2", chains=["JAVELIN"])
c.plotter.plot_contour(ax[1], "lags_1","lags_2", chains=["LITMUS"])

plt.tight_layout()

#=======================================
#COMPARISON

c2 = ChainConsumer()

fname = sim+"-"+os.path.basename(os.getcwd())

c2.add_chain([DATA_JAVELIN['lags_1'],DATA_JAVELIN['lags_2']],  parameters = labels, name = "JAVELIN", color='purple')
c2.add_chain([DATA[:,0],DATA[:,1]],                            parameters = labels, name = "LITMUS",  color='blue')

fig2_figsize = (3,3)
fig2_truth   = [truth['lags_1'],truth['lags_2']]
fig2_extents = [extents['lags_1'],extents['lags_2']]
fig2_display = True

c2.plotter.plot(chains = ["JAVELIN"], truth=fig2_truth, extents=fig2_extents, figsize=fig2_figsize, filename = fname+"-JAVELIN")
plt.tight_layout()

c2.plotter.plot(chains = ["LITMUS"], truth=fig2_truth, extents=fig2_extents, figsize=fig2_figsize, filename = fname+"-LITMUS")
plt.tight_layout()

#plt.show()
