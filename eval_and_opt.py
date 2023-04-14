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
    '''
    #Safety check to make sure we don't over-write when using fixed_params
    fixed_params= copy(fixed_params)
    for key in scan_params.keys():
        if key in fixed_params.keys():
            fixed_params.pop(key)



#=================================================================
if __name__=="__main__":
    from data_utils import array_to_lc, lc_to_banded, data_tform, normalize_tform, flatten_dict

    print("Starting tests for fitting procedure")

    # load some example data
    cont  = array_to_lc(np.loadtxt("./Data/data_fake/360day/cont.dat"))
    line1 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line1.dat"))
    line2 = array_to_lc(np.loadtxt("./Data/data_fake/360day/line2.dat"))

    #Make into banded format
    banded_2line  = lc_to_banded([cont, line1, line2])
    banded_1line  = lc_to_banded([cont, line1])
    banded_cont   = lc_to_banded([cont])

    #Fire off a short MCMC run
    MCMC_params={
        "Nchain": 1,
        "Nburn": 10,
        "Nsample": 20,
        }

    for band in [banded_2line, banded_1line, banded_cont]:
        print("Unit tests for %i lines: " %max(band["bands"]) )
        if max(band['bands']!=0):
            print("log prob for  lag1, lag2 =0:\t%f"    % loss(band, params = {"lags": jnp.array([0,0])}))
            print("log prob for  lag1, lag2 =360:\t%f"  % loss(band, params={"lags": jnp.array([360, 360])}))

    print("unit tests succesfful")

