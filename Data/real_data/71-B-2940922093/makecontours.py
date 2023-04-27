
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pylab as plt

SIGNAL = np.loadtxt("banded_data.dat")

#==================================

DATA = np.loadtxt("outchain.dat")
NEST_SEED = np.loadtxt("outchain-nest-seed.dat")
NEST_FULL = np.loadtxt("outchain-nest-full.dat")
KEYS = np.loadtxt("outchain_keys.dat", dtype="str_")

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
#DOUBLE LINE FITS

c = ChainConsumer()
c.add_chain(DATA, parameters = list(KEYS), name = "HMC")
c.add_chain(NEST_SEED, parameters = list(KEYS), name = "Nest Seeds")
c.add_chain(NEST_FULL, parameters = list(KEYS), name = "Nest Full")

c.plotter.plot(filename ="./contours.png", extents=extents)
c.plotter.plot(filename ="./contours_lagsonly.png", extents=extents, parameters = ["lags_1","lags_2"])

c.remove_chain()
c.remove_chain()

c.plotter.plot(filename ="./contours_clean.png", extents=extents)
c.plotter.plot(filename ="./contours_clean_lagsonly.png", extents=extents, parameters = ["lags_1","lags_2"])
c.plotter.plot(filename ="./contours_clean_contonly.png", extents=extents, parameters = ["log_sigma_c","log_tau"])
c.plotter.plot_distributions(filename = "./summary_clean.png", extents=extents)

#=======================================
#SINGLE LINE COMPARISON

DATA_1 = np.loadtxt("outchain-line1.dat")
DATA_2 = np.loadtxt("outchain-line2.dat")

c2 = ChainConsumer()
c2.add_chain(DATA[:,:2], parameters = ["lags_1","lags_2"], name = "Simultaneous Fitting")
c2.add_chain(np.array([DATA_1[:,0],DATA_2[:,0]]).T, parameters = ["lags_1","lags_2"], name = "Simultaneous Fitting")
c2.plotter.plot(filename ="./contours_linecomparison.png", extents=extents)
