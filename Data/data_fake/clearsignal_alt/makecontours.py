from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pylab as plt



#==================================

SIGNAL = np.loadtxt("banded_data.dat")
DATA = np.loadtxt("outchain.dat")
KEYS = np.loadtxt("outchain_keys.dat", dtype="str_")

Nbands = (DATA.shape[1]-3) / 3 + 1

inds = np.ones(DATA.shape[1])
print("Vars /w no changes:")
for column in range(DATA.shape[1]):
    
    if len(np.unique(DATA[:,column]))==1:
        print("\t"+KEYS[column])
        inds[column]=False
inds = np.where(inds==True)[0]
DATA=DATA[:,inds]
KEYS = KEYS[inds]

c = ChainConsumer()
c.add_chain(DATA, parameters = list(KEYS))

if Nbands==3:
    truth = {"rel_amps_0" : 1,
             "rel_amps_1" : 1,
             "log_sigma_c" : 0,
             "log_tau" : 5.99,
             "means_0":0,
             "means_1":0,
             "means_2":0}

    extents = {"rel_amps_0" :       (0,10),
             "rel_amps_1" :         (0,10),
             "log_sigma_c" :    (-2.3,2.3),
             "log_tau" :        (2,8),
             "means_0":         (-10,10),
             "means_1":         (-10,10),
             "means_2":         (-10,10)
               }
elif Nbands==2:
    truth = {"rel_amps" : 1,
             "log_sigma_c" : 0,
             "log_tau" : 5.99,
             "means_0":0,
             "means_1":0,
             }


    extents = {"rel_amps" :       (0,10),
             "log_sigma_c" :    (-2.3,2.3),
             "log_tau" :        (2,8),
             "means_0":         (-10,10),
             "means_1":         (-10,10),
               }
elif Nbands == 1:
    truth = {
             "log_sigma_c" : 0,
             "log_tau" : 5.99,
             "mean":0,
             }


    extents = {
             "log_sigma_c" :    (-2.3,2.3),
             "log_tau" :        (2,8),
             "mean":         (-10,10),
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
