import sys
import os

sys.path.append('..')

from data_utils import lc_to_banded
from eval_and_opt import realize, interp

import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------
scale = 400
sigma_c = 0.5
X = np.array([0,1, 2, 3, 4, 5]) * scale
Y = np.array([0, 1.0, 0.8, -0.4, 0.2, -1.0])
E = np.random.rand(len(X)) * 0.2 + 0.2

n_reals = 10
Tplot = np.linspace(-2,10,256) * scale
#-------------------------------------

data = {'T':X, "Y":Y, 'E':E}
data = lc_to_banded([data])

#-------------------------------------

plt.figure(figsize=(6,2))
label = 'Realizations'
for i in range(32):
    Y_out = realize(data, Tout = Tplot, seed=i, params = {'log_tau': np.log(scale), 'log_sigma_c': np.log(sigma_c)})
    plt.plot(Tplot,Y_out, lw=0.5, c='k', alpha=0.1, label = label)
    label = None

plt.errorbar(X,Y,yerr=E, fmt='none', capsize = 5, c = 'dodgerblue', label = 'Measurements')

Y_interp, E_interp = interp(data, Tout=Tplot, params={'log_tau': np.log(scale), 'log_sigma_c': np.log(sigma_c)})
E_interp = np.sqrt(E_interp)
plt.plot(Tplot,Y_interp, c= 'k', lw=2)
plt.plot(Tplot,Y_interp + E_interp, c = 'k', lw=1, ls='--')
plt.plot(Tplot,Y_interp - E_interp, c = 'k', lw=1, ls='--')

plt.axhline(sigma_c,c='r',lw=1)
plt.axhline(-sigma_c,c='r',lw=1)
plt.axhline(0,ls='--',c='k')

plt.xlim(min(Tplot),max(Tplot))

plt.xlabel("Time")
plt.ylabel('Signal')
plt.legend(loc='lower right')
plt.ylim(-sigma_c*3,sigma_c*3)
plt.tight_layout()
plt.show()

print("Done")