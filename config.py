'''
config.py

Contains any ranges or start locations that are shared across many files.
Stored here for easy access and consistency

'''
#================================================

#----------------------------------
#Model search ranges
lag_min = 0.0
lag_max = 800

log_tau_min = 1.0
log_tau_max = 8.0

log_sigma_c_min = -2.5
log_sigma_c_max = 2.5

rel_amp_min =0.0
rel_amp_max =10.0

mean_min = -20.0
mean_max = 20.0

#----------------------------------
# Default extents

plot_extents = {
    "log_sigma_c": [log_sigma_c_min, log_sigma_c_max],
    "log_tau": [log_tau_min, log_tau_max],
    "lags_1": [lag_min, lag_max],
    "lags_2": [lag_min, lag_max],
    "rel_amps_1": [lag_min, rel_amp_max],
    "rel_amps_2": [lag_min, rel_amp_max],
    "means_0": [mean_min, mean_max],
    "means_1": [mean_min, mean_max],
    "means_2": [mean_min, mean_max],
}


#----------------------------------
# 'decent fit paramaters'
start_params = {
    "log_sigma_c": 0,
    "log_tau": 6,
    "lags_1": 0,
    "lags_2": 0,
    "rel_amps_1": 1,
    "rel_amps_2": 1,
    "means_0": 0,
    "means_1": 0,
    "means_2": 0,
}

#----------------------------------
# MCMC default values

default_MCMC_params={
    "Ncores": 1,
    "Nchain": 300,
    "Nburn": 200,
    "Nsample": 600,

    "step_size": 1E-3,
    "progress_bar": True,
    "targ_acc_prob": 0.9,

    "ns_num_live":0,
    "ns_max_samples":0
}