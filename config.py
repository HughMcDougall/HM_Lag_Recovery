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

log_tau_min = 3.0
log_tau_max = 10.0

log_sigma_c_min = -2.5
log_sigma_c_max = 2.5

rel_amp_min =0.0
rel_amp_max =10.0

mean_min = -10.0
mean_max = 10.0

#----------------------------------
# Default seed values
'''
[MISSINGNO]
'''


#----------------------------------
# MCMC default values

default_MCMC_params={
    "Ncores": 1,
    "Nchain": 300,
    "Nburn": 200,
    "Nsample": 600,
    "step_size": 1E-3,
    "progress_bar": True,

    "ns-num_live":0,
    "ns-max_samples":0
}