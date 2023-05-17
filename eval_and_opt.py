from copy import deepcopy as copy
import jax
import jax.numpy as jnp
import data_utils
from modelling import build_gp
import numpy as np

# ============================================
@jax.jit
def _loss_nline(data, params, Nbands):
    '''
    Calculates the log likelihood for a set of data and params in a jit friendly form
    '''

    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    Nbands = jnp.max(bands)+1

    # ----------------------------------
    # Extract params

    log_tau     = params["log_tau"]
    log_sigma_c = params["log_sigma_c"]
    means       = params["means"]

    lags        = params["lags"]
    rel_amps    = params["rel_amps"]
    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.concatenate([jnp.array([1]), rel_amps]) * jnp.exp(log_sigma_c)


    gp_params  = {"tau:": tau,
                  "lags": jnp.concatenate([jnp.array([1]), lags]),
                  "amps": amps,
                  "means:": means}

    # ----------------------------------
    # Apply lags and sort data
    T -= gp_params['lags'][bands]

    inds = jnp.argsort(T)

    T       =   T[inds]
    Y       =   Y[inds]
    E       =   E[inds]
    bands   =   bands[inds]
    # ----------------------------------
    # Build and sample GP
    # Build TinyGP Process
    gp = build_gp(T=T, Y=Y, diag=E*E, bands=bands, tau=tau, amps=amps, means=means)

    # Apply likelihood
    return gp.log_probability(Y)

@jax.jit
def _loss_cont(data, params):
    '''
    Calculates the log likelihood for a set of data and params in a jit friendly form
    '''

    data = copy(data)
    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    Nbands = jnp.max(bands)+1

    # ----------------------------------
    # Extract params

    log_tau     = params["log_tau"]
    log_sigma_c = params["log_sigma_c"]
    means       = params["means"]
    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.array([jnp.exp(log_sigma_c)])


    gp_params  = {"tau:": tau,
                  "amps": amps,
                  "means:": means}

    # ----------------------------------
    # Build and sample GP
    # Build TinyGP Process
    gp = build_gp(T=T, Y=Y, diag=E*E, bands=bands, tau=tau, amps=amps, means=means)

    # Apply likelihood
    return gp.log_probability(Y)


def loss(data, params):
    Nbands = jnp.max(data['bands']) + 1
    if Nbands>1:
        params = data_utils.default_params(Nbands) | params
        return (_loss_nline(data, params, Nbands))
    else:
        params = data_utils.default_params(Nbands) | params
        return (_loss_cont(data, params))

#=================================================================
def scan(banded_data, scan_params, fixed_params = "None"):
    '''
    Runs a scan over
    :param scan_params:
    :param fixed_params:
    :return:

    [MISSINGNO] - When finished, this will automatically evaluate 'loss' over certain axis
    '''
    #Safety check to make sure we don't over-write when using fixed_params
    fixed_params= copy(fixed_params)
    for key in scan_params.keys():
        if key in fixed_params.keys():
            fixed_params.pop(key)

#=================================================================

def realize(data, params, Tout, band=0, nreals=1, seed=0):
    '''
    :param data:
    :param params:
    :param Tout:
    :return:

    Generates a conditioned realization for a set of data
    '''

    data = copy(data)

    T, Y, E, bands = data['T'], data['Y'], data['E'], data['bands']
    Nbands = jnp.max(bands)+1

    params = copy(params)
    params = data_utils.default_params(Nbands) | params

    # ----------------------------------
    # Extract params

    log_tau     = params["log_tau"]
    log_sigma_c = params["log_sigma_c"]
    means       = params["means"]

    lags        = params["lags"]
    rel_amps    = params["rel_amps"]
    # ----------------------------------
    # Collect params for sending to GP
    tau  = jnp.exp(log_tau)
    amps = jnp.concatenate([jnp.array([1]), rel_amps]) * jnp.exp(log_sigma_c)


    gp_params  = {"tau:": tau,
                  "lags": jnp.concatenate([jnp.array([1]), lags]),
                  "amps": amps,
                  "means:": means}

    # ----------------------------------
    # Apply lags and sort data
    T -= gp_params['lags'][bands]

    inds = jnp.argsort(T)

    T       =   T[inds]
    Y       =   Y[inds]
    E       =   E[inds]
    bands   =   bands[inds]
    # ----------------------------------
    # Build TinyGP Process
    gp = build_gp(T=T, Y=Y, diag=E*E, bands=bands, tau=tau, amps=amps, means=means)

    # Condition GP against data
    gp_conditoned =gp.condition(Y, (Tout, jnp.zeros_like(Tout, dtype=int) + band))

    # ----------------------------------
    # Calculate and return

    Yout = gp_conditoned.gp.sample(jax.random.PRNGKey(seed), shape=(nreals,))

    if nreals==1:
        return(Yout[0,:])
    else:
        return (Yout)


#=================================================================
if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict
    import matplotlib.pylab as plt



    print("Starting tests for fitting procedure")

    # load some example data
    # targ = "./Data/data_fake/clearsignal/"
    targ = "./Data/data_fake/150day-good/"
    #targ = "./Data/data_fake/360day/"

    cont  = array_to_lc(np.loadtxt(targ+"cont.dat"))
    line1 = array_to_lc(np.loadtxt(targ+"line1.dat"))
    line2 = array_to_lc(np.loadtxt(targ+"line2.dat"))

    lag1 = 150
    lag2 = 160

    #Make into banded format
    banded_2line  = lc_to_banded([cont, line1, line2])
    banded_1line  = lc_to_banded([cont, line1])
    banded_cont   = lc_to_banded([cont])

    for band in [banded_2line, banded_1line, banded_cont]:
        print("Unit tests for %i lines: " %max(band["bands"]) )
        if max(band['bands']!=0):
            print("log prob for  lag1, lag2 =0:\t%f"    % loss(band, params = {"lags": jnp.array([0,0])}))
            print("log prob for  lag1, lag2 =%i:\t%f"  %(lag2, loss(band, params={"lags": jnp.array([lag1, lag2])})))


    #Realization Generation
    print("Doing several realizations")

    plt.figure()
    Tgrid = np.linspace(    0,  np.max(banded_2line['T'])    , 1024)

    plt.errorbar(cont['T'], cont['Y'], fmt='none', yerr=cont['E'], capsize=4, c='b')
    plt.errorbar(line1['T']-lag1, line1['Y'], fmt='none', yerr=line1['E'], capsize=4, c='r')
    plt.errorbar(line2['T']-lag2, line2['Y'], fmt='none', yerr=line2['E'], capsize=4, c='g')

    plt.scatter(cont['T'], cont['Y'], c='b')
    plt.scatter(line1['T']-lag1, line1['Y'], c='r')
    plt.scatter(line2['T']-lag2, line2['Y'], c='g')

    for i in range(10):
        print(i)

        yplot = realize(data = banded_2line,
                        params = {"lags": jnp.array([lag1,lag2]),
                                  'means': jnp.array([0,0,0]),
                                  'log_tau': np.log(400)},
                        Tout = Tgrid ,
                        seed=i, band=1)

        plt.plot(Tgrid,yplot, alpha= 0.1, c='k')

    plt.ylim(-5,5)
    plt.show()

    print("unit tests succesfful")

