import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy as copy
from chainconsumer import ChainConsumer

#-------------------------------------------------------------------------
K = 1000
a = 2
nsamples = 800
nburn = 700
n=2

sig1 = 60/2
sig2 = 60/2
sep  = 360

def P(X):
    x,y = X
    out = np.exp( -1/2 * (x/sig1)**2) * (
                                           np.exp(-1 / 2 * ((y - sep/2) / sig2) ** 2)
                                         + np.exp(-1 / 2 * ((y + sep/2) / sig2) ** 2)
                                         )
    out/=sig1*np.sqrt(2)*np.pi
    return(out)

def Pgrid(X):
    x,y = X
    out = np.exp( -1/2 * (x/sig1)**2) * (
                                           np.exp(-1 / 2 * ((y - sep/2) / sig2) ** 2)
                                         + np.exp(-1 / 2 * ((y + sep/2) / sig2) ** 2)
                                         + np.exp(-1 / 2 * ((y - sep*3/2)   / sig2) ** 2)
                                         + np.exp(-1 / 2 * ((y + sep*3/2)   / sig2) ** 2)
                                         )
    out/=np.sqrt(sig1*sig2) * 2*np.pi
    return(out)

def rand_z():
    while True:
        r = np.random.rand()
        z = np.random.rand()*(a-1/a) + 1/a
        g = (z/a)**(-1/2)
        if r<g: return(z)

#-------------------------------------------------------------------------
X1 = np.random.normal(size=(K//2,2),loc=(0,sep/2),scale=(sig1,sig2))
X2 = np.random.normal(size=(K-len(X1),2),loc=(0,-sep/2),scale=(sig1,sig2))
Xstart = np.concatenate([X1,X2])
np.random.shuffle(Xstart)

X = copy(Xstart)

Xout  = np.zeros([(nsamples+1)*K,2])
Xout[:K] = X

new = 0
for i in range(1,nsamples+1):
    if i%100==0: print(i)
    Xnew = copy(X)
    for k in range(K):
        #index in Xout to save to
        j = i*K + k


        k_comp = np.random.randint(K)
        Xmove = X[k]
        Xcomp = X[k_comp]

        z = rand_z()
        Xprop = Xmove * z + Xcomp * (1-z)

        r = np.random.rand()
        if min(1,P(Xprop) / P(Xmove) * z**(n-1)) > r:
            X[k] = Xprop
            Xout[j] = Xprop
            Xnew[k] = Xprop
            new+=1
        else:
            Xout[j] = Xmove
            Xnew[k] = Xmove
    X = copy(Xnew)
print(new / (K*nsamples))


#-------------------------------------------------------------------------


Xchain = [np.array([Xout[i*K+k,:] for i in range(nsamples+1)]) for k in range(K)]
for chain in Xchain[::50]:
    plt.plot(chain[:,0],chain[:,1],lw=0.1)

plt.axis('equal')

plt.scatter(Xout[:K,0],Xout[:K,1],s=0.5)
plt.scatter(Xout[-K:,0],Xout[-K:,1],s=0.5)
plt.ylim([-sep*2,sep*2])
plt.show()

pltlims = sep*2
fig,ax = plt.subplots(2,1,sharex=True)
ax[0].hist(Xstart[:,1],bins=64, range=[-pltlims , pltlims ], density = True)
ax[1].hist(Xout[nburn*K:,1],bins=64, range=[-pltlims , pltlims ], density = True)

ys = np.linspace(-pltlims ,pltlims ,256)
ax[0].plot(ys,[P([0,y]) for y in ys])
ax[1].plot(ys,[P([0,y]) for y in ys])
plt.show()

c = ChainConsumer()
c.add_chain(Xstart)
c.add_chain(Xout[nburn*K:])
c.plotter.plot()
plt.tight_layout()
plt.show()