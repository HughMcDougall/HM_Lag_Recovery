
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pylab as plt
#==================================

print("Loading LITMUS Data")
DATA_LITMUS = np.loadtxt("outchain.dat")
NEST_SEED = np.loadtxt("outchain-nest-seed.dat")
NEST_FULL = np.loadtxt("outchain-nest-full.dat")

KEYS_LITMUS = np.loadtxt("outchain_keys.dat", dtype="str_")

DATA_LITMUS = {key:DATA_LITMUS[:,i] for key,i in zip(KEYS_LITMUS, range(len(KEYS_LITMUS)))}

print("Loading JAVELIN Data")
DATA_JAVELIN = np.loadtxt("chain-twoline-highqual.dat")
DATA_JAVELIN = {
    "lags_1": DATA_JAVELIN[:,2],
    "lags_2": DATA_JAVELIN[:,5]
    }
DATA_JAVELIN_LINES = {
        "lags_1": np.loadtxt("chain-line1-highqual.dat")[:,2],
        "lags_2": np.loadtxt("chain-line2-highqual.dat")[:,2]
        }

#=======================================
extents = {"amps_0" :       (0,10),
         "amps_1" :         (0,10),
         "log_sigma_c" :    (-2.5,2.5),
         "log_tau" :        (1,8),
         "means_0":         (-20,20),
         "means_1":         (-20,20),
         "means_2":         (-20,20),
         "lags_1":          (0,800),
         "lags_2":          (0,800)
           }

#=======================================
print("Loading doing LITMUS contours")
c = ChainConsumer()
c.add_chain(DATA_LITMUS, name = "HMC")
c.add_chain(NEST_SEED, parameters = list(KEYS_LITMUS), name = "Nest Seeds")
c.add_chain(NEST_FULL, parameters = list(KEYS_LITMUS), name = "Nest Full")

c.plotter.plot(filename ="./contours.png", extents=extents)
c.plotter.plot(filename ="./contours_lagsonly.png", extents=extents, parameters = ["lags_1","lags_2"])

c.remove_chain()
c.remove_chain()

c.plotter.plot(filename ="./contours_clean.png", extents=extents)
c.plotter.plot(filename ="./contours_clean_lagsonly.png", extents=extents, parameters = ["lags_1","lags_2"])

#=======================================

print("Loading doing JAVELIN contours")
cjav = ChainConsumer()
cjav.add_chain(DATA_JAVELIN, name = "Simultaneous Line Fitting", color='purple')
cjav.add_chain(DATA_JAVELIN_LINES, name = "Independent Line Fitting", color='cyan')
cjav.plotter.plot(filename ="./contours_lagsonly_JAVELIN.png", extents=extents)

#=======================================
print("Loading doing Chain comparisons")
nchains = 300
s_jav       = len(DATA_JAVELIN["lags_1"]) // nchains
s_litmus    = len(DATA_LITMUS["lags_1"]) // nchains
Xs_JAV = [DATA_JAVELIN["lags_1"][s_jav*i:s_jav*(i+1)] for i in range(nchains)]
Ys_JAV = [DATA_JAVELIN["lags_2"][s_jav*i:s_jav*(i+1)] for i in range(nchains)]
Xs_JAV_LINES = [DATA_JAVELIN_LINES["lags_1"][s_jav*i:s_jav*(i+1)] for i in range(nchains)]
Ys_JAV_LINES = [DATA_JAVELIN_LINES["lags_2"][s_jav*i:s_jav*(i+1)] for i in range(nchains)]
Xs_LITMUS = [DATA_LITMUS["lags_1"][s_litmus*i:s_litmus*(i+1)] for i in range(nchains)]
Ys_LITMUS = [DATA_LITMUS["lags_2"][s_litmus*i:s_litmus*(i+1)] for i in range(nchains)]

fig,ax=plt.subplots(1,3,figsize=(15,5))

for i in range(nchains):
    ax[0].plot(Xs_JAV[i],       Ys_JAV[i],          lw=0.1)
    ax[1].plot(Xs_JAV_LINES[i], Ys_JAV_LINES[i],    lw=0.1)
    ax[2].plot(Xs_LITMUS[i],    Ys_LITMUS[i],       lw=0.1)

ax[0].set_title("JAVELIN, simultanteous fitting")
ax[1].set_title("JAVELIN, independent fitting")
ax[2].set_title("LITMUS")

ax[0].set_xlim(extents["lags_1"])
ax[0].set_ylim(extents["lags_2"])
ax[1].set_xlim(extents["lags_1"])
ax[1].set_ylim(extents["lags_2"])
ax[2].set_xlim(extents["lags_1"])
ax[2].set_ylim(extents["lags_2"])
plt.tight_layout()
plt.savefig(fname = "chain_comparison.png", format = "png")

print("All plots Done")
