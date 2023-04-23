import sys

sys.path.append("..")

import numpy as np
import matplotlib.pylab as plt
from data_utils import array_to_lc

import numpy as np
import matplotlib.pylab as plt
from numpy import dot
from numpy.linalg import inv as inv
from scipy.stats import poisson
from copy import deepcopy as copy


def covar(T, T2=None, tau=1):
    '''
    Make covariance matrix
    '''

    if type(T2) == type(None): T2 = T

    A, B = np.meshgrid(T, T2)

    K = np.exp(-abs(A - B) / tau)

    return (K.T)


def mean_and_var(T,Y,E,n):
    #-------------------------------------------------

    T_out = np.linspace(tmin, tmax, n)

    K = covar(T, T, tau)
    K_ = covar(T, T_out, tau)
    K__ = covar(T_out, T_out, tau)

    m = np.mean(Y)

    diag = np.diag(E)
    K = K + diag

    Y_out = dot(K_.T, dot(inv(K), Y - m)) + m
    C_ = K__ - dot(K_.T, dot(inv(K), K_))
    E_out = np.diag(C_) ** 0.5
    '''

    A=1/tau
    VARINV = np.zeros(n)
    Y_out = np.zeros(n)
    for i in range(len(T)):
        varinv=(A*abs(T_out - T[i])+E[i]**2)**-1
        VARINV+=varinv
        Y_out +=Y[i]*varinv
    Y_out/=VARINV
    E_out = VARINV**-(1/2)
    
    '''
    return(T_out,Y_out,E_out)


def likelihood(Y, K):
    '''
    Likelihood function
    '''

    y = Y
    out = 1 / np.sqrt(np.linalg.det(K))
    out *= np.exp(-1/2* dot(y, dot(inv(K), y)) )
    return (out)


#================================



T = np.array([0,1,2,3,4,5,6,7])
Y = np.array([0.0582586196482984,0.151872188832674,-0.148962675924955,0.207374188878444,0.0238493457315756,0.331763743781315,0.214770784789552,0.0582586196482988,])
E = np.array([1.38152734314848,1.25017457648278,1.40631378045828,1.32955843505642,1.41759165796199,1.07202966109027,1.0179426464457,1.05232195545219,])


T1 = T[::2] + T[1]
Y1 = Y[::2]
E1 = E[::2]

T2 = T[1::2]
Y2 = Y[1::2]
E2 = E[1::2]

np.random.seed(3)
#Y2 = np.roll(Y2,3)
Y2+=np.random.randn(len(Y2))
np.random.shuffle(Y2)

# load some example data
tau = 100
tmin,tmax = 0,2500
ymin,ymax = -4,4

x_n = 512
y_n = 512

T1 = T1*tmax*0.75 / np.max(T)
T2 = T2*tmax*0.75 / np.max(T)
E1 = E1 / 4
E2 = E2 / 4


#-------------------------------------
Tshift,Yshift,Eshift = T2+(T1[1]-T1[0])/2, copy(Y2), copy(E2)

Tplot, Yplot1, Eplot1 = mean_and_var(T1,Y1,E1,x_n)
Tplot, Yplot2, Eplot2 = mean_and_var(T2,Y2,E2,x_n)
Tplot, Yplot3, Eplot3 = mean_and_var(Tshift,Yshift,Eshift,x_n)

#----------------------------------------------
y_forgrid = np.linspace(ymin,ymax,y_n)
Xgrid, Ygrid = np.meshgrid(Tplot,y_forgrid)

Z1 = np.zeros([x_n,y_n])
Z2 = np.zeros([x_n,y_n])
Z3 = np.zeros([x_n,y_n])
for i in range(x_n):
    for j in range(y_n):
        t,y = Tplot[i],y_forgrid[j]
        Z1[i,j]  = np.exp( - ((y - Yplot1[i]) / Eplot1[i])**2 / 2) / Eplot1[i] / np.sqrt(2 *np.pi)
        Z2[i, j] = np.exp(- ((y - Yplot2[i]) / Eplot2[i]) ** 2 /2) / Eplot2[i] / np.sqrt(2 *np.pi)
        Z3[i, j] = np.exp(- ((y - Yplot3[i]) / Eplot3[i]) ** 2 /2) / Eplot3[i] / np.sqrt(2 *np.pi)
Z1 = Z1[:,::-1].T
Z2 = Z2[:,::-1].T
Z3 = Z3[:,::-1].T


#=======================================
plt.figure(figsize=(4, 2.0))

plt.imshow(Z1, aspect="auto",extent=[tmin,tmax,ymin,ymax], cmap='binary')
plt.errorbar(T1,Y1,yerr=E1*1.96,fmt="none",capsize=2,label="Measurements")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.0, 0, 0))
plt.tight_layout()
plt.show()

T_m2 = T2
Y_m2 = Y2
E_m2 = E2

Y_2 = Yplot2
E_2 = Eplot2
Z_2 = Z2


for i in range(2):

    Ediff = (Eplot1**2 + E_2**2)**(1/2)
    Ebest = (Eplot1 ** -2 + E_2 ** -2) ** (-1 / 2)
    Ybest = (Yplot1*Eplot1**-2 + Y_2 * E_2**-2) / (Eplot1**-2 + E_2**-2)
    dY = (Yplot1-Y_2)
    badness = np.exp(-((Yplot1-Ybest)/Ebest)**2 + -((Y_2-Ybest)/Ebest)**2)

    badness  = np.exp(-(dY/Eplot1)**2)


    Z = (Z1 / np.max(Z1))**1.5

    alph = 2
    bet = 1.4
    gam = 0.25
    R = 1-Z*alph*(badness * (1-gam) + gam)**bet
    G = (1-Z*alph )
    B = (1-Z*alph)


    C = np.stack([R,G,B],axis=2)

    plt.figure(figsize = (5,2.0))
    plt.imshow(C, aspect="auto",extent=[tmin,tmax,ymin,ymax], cmap='binary', vmin=0, vmax=1)

    plt.plot(Tplot,Yplot1, c='green')
    plt.plot(Tplot,Y_2, c='orange')
    plt.errorbar(T1,Y1,yerr=E1*1.96,fmt="none",capsize=2, label = "Measurement Series 1", c='green')
    plt.errorbar(T_m2,Y_m2,yerr=E2*1.96,fmt="none",capsize=2,label = "Measurement Series 2", c='orange')
    plt.ylim(ymin,ymax)

    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3, 0, 0),ncols = 2)
    plt.tight_layout()
    plt.show()


    Z_2 = Z3

    T_m2 = Tshift
    Y_m2 = Yshift
    E_m2 = Eshift

    Y_2 = Yplot3
    E_2 = Eplot3


lags = np.linspace(0,tmax,1024)
likelihoods=np.zeros(1024)

Y = np.concatenate([Y1,Y2])
E = np.concatenate([E1,E2])

for i in range(len(lags)):
    lag = lags[i]

    T = np.concatenate([T1,T2+lag])
    K = covar(T,tau=tau) + np.diag(E*E)

    likelihoods[i] = likelihood(Y,K)
likelihoods/=np.max(likelihoods)

plt.figure(figsize=(5,2.5))
plt.plot(lags,np.log(likelihoods))
dt = T1[1] - T1[0]
for i in range(int(max(lags)//dt)+1):

    if i ==0:
        plt.axvspan(dt * i + tau, dt * (i + 1) - tau, alpha=0.1, color='green', label="Aliasing Peak")
        plt.axvline(dt*i,c='k',ls='-',label="On-Season Shift")
        plt.axvline(dt*i+tau,c='k',ls=':',label="$+/- \\tau$",lw=0.5)
        plt.axvline(dt*i-tau,c='k',ls=':',lw=0.5)
    else:
        plt.axvspan(dt * i + tau, dt * (i + 1) - tau, alpha=0.1, color='green')
        plt.axvline(dt * i, c='k', ls='-')
        plt.axvline(dt*i+tau,c='k',ls=':',lw=0.5,)
        plt.axvline(dt*i-tau,c='k',ls=':',lw=0.5)
plt.axhline(0,c='k',lw=0.5,ls='--')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols = 3)
plt.ylim(-5,1)
plt.xlabel("Time-Shift Of Signal 2")
plt.ylabel("$\Delta ln|\mathcal{L}| $")
plt.xlim(-tau,tmax)
plt.tight_layout()

plt.show()