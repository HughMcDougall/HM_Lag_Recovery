
from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pylab as plt
import tinygp

import os
import sys
sys.path.append("....")
sys.path.append("C:/Users/hughm/My Drive/HonoursThesis/HM_Lag_Recovery")
import 
#==================================

#==================================

SIGNAL = np.loadtxt("banded_data.dat")

try:
    DATA = np.loadtxt("outchain.dat")
    NEST_SEED = np.loadtxt("outchain-nest-seed.dat")
    NEST_FULL = np.loadtxt("outchain-nest-full.dat")
    KEYS = np.loadtxt("outchain_keys.dat", dtype="str_")
except:
    
    DATA = np.loadtxt("outchain-twoline.dat")
    NEST_SEED = np.loadtxt("outchain-twoline-nest-seed.dat")
    NEST_FULL = np.loadtxt("outchain-twoline-nest-full.dat")
    KEYS = np.loadtxt("outchain-twoline_keys.dat", dtype="str_")

#==================================
