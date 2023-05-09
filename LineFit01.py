'''
#============================================
LineFit01.py

Slurm-friendly runtime to be triggered in batches

HM 3/4/23
#============================================
'''


from argparse import ArgumentParser
import numpy as np

import SIMBA
from fitting_procedure import fit_single_source
from data_utils import lc_to_banded, flatten_dict, array_to_lc, normalize_tform, data_tform


#============================================
#Main Runtime Test
def main():
    '''
    From command line, takes:
        -N         <int>    jax device count. Should be equal to chains (Redundant?)
        -i         <int>    which job in the table to run
        -Nchain    <int>    Nchain
        -Nburn     <int>    Nburn
        -Nsample   <int>    Nsamples
        -step_size <float>  HMC step size
    '''

    #=======================
    #COMMAND LINE ARGUMENTS
    ap = ArgumentParser(description='Reverberation mapping with numpyro')

    ap.add_argument('-Ncores', '--Ncores',              metavar='Ncores',       type=int,       help='Number of devices to feed to jax',    default=1)
    ap.add_argument('-i', '--i',                        metavar='i',            type=int,       help='job itteration number',               default=0)
    ap.add_argument('-Nchains', '--Nchains',            metavar='Nchains',      type=int,       help='Number of MCMC chains',               default=0)
    ap.add_argument('-Nburn', '--Nburn',                metavar='Nburn',        type=int,       help='Number of burn-in steps',             default=200)
    ap.add_argument('-Nsamples', '--Nsamples',          metavar='Nsamples',     type=int,       help='Number of samples',                   default=200)
    ap.add_argument('-NS_numlive', '--NS_numlive',      metavar='NS_numlive',   type=int,       help='Number of live points in NS',         default=0)
    ap.add_argument('-NS_maxevals', '--NS_maxevals',    metavar='NS_maxevals',  type=int,       help='Max samples in NS',                   default=0)
    ap.add_argument('-step_size', '--step_size',        metavar='step_size',    type=float,     help='Step Size in HMC',                    default=1E-3)
    ap.add_argument('-progress_bar', '--progress_bar',  metavar='progress_bar', type=float,     help='numpyro progress bar',                default=False)
    ap.add_argument('-table', '--table',                metavar='table',        type=str,       help='SIMBA table to draw jobs from',       default=SIMBA._def_tab_url)
    ap.add_argument('-targ_acc', '--targ_acc',          metavar='targ_acc',     type=float,     help='targ_accept prob in NUTS burnin',     default=0.9)

    args = ap.parse_args()
    job_args = SIMBA.get_args(args.i, table_url=args.table)

    #=======================
    # PRINTING
    print("JOB STARTING")
    print("---------------------------------")
    print("Input urls:")
    print("\t",job_args["cont_url"])
    print("\t",job_args["line1_url"])
    print("\t",job_args["line2_url"])
    #=======================
    # GET AND NORMALIZE DATA

    #Get targets
    cont  = array_to_lc(np.loadtxt(job_args["cont_url"]))
    line1 = array_to_lc(np.loadtxt(job_args["line1_url"]))
    line2 = array_to_lc(np.loadtxt(job_args["line2_url"]))

    try:
        mode = job_args["mode"]
    except:
        mode = "twoline"
    assert mode in ['twoline', "line1", "line2", "cont"], "bad mode input %s in LineFit01.py job %i" %(mode,job_args["i"])

    if mode == "twoline": lcs_unbanded = [cont, line1, line2]
    elif mode == "line1": lcs_unbanded = [cont, line1]
    elif mode == "line2": lcs_unbanded = [cont, line2]
    elif mode == "cont": lcs_unbanded = [cont]

    print("Beginning job %i on job list %s with mode %s" % (args.i, args.table, mode))

    # Convert to banded and normalize
    banded_data = lc_to_banded(lcs_unbanded)
    banded_data = data_tform(banded_data, normalize_tform(banded_data))

    # Output light curve for checking
    if mode == "twoline":
        out, out_keys = flatten_dict(banded_data)
        print("Data loaded. Saving normalized lightcurve to %s" %(job_args["out_url"]+"banded_data.dat"))
        try:
            np.savetxt(job_args["out_url"]+"banded_data.dat", out)
        except:
            print("Unable to locate output folder %s. Saving to local runtime folder instead" %job_args["out_url"])
            np.savetxt("./banded_data.dat",out)
            
    #=======================
    # PERFORM FITTING
    nsamples= 600
    if mode!="twoline": nsamples = 300

    MCMC_params ={
        "Ncores":           args.Ncores,
        "Nchain":           args.Nchains,
        "Nburn":            args.Nburn,
        "Nsample":          nsamples,

        "step_size":        args.step_size,
        "progress_bar":     args.progress_bar,
        "targ_acc_prob":    args.targ_acc,

        "ns_num_live":      args.NS_numlive,
        "ns_max_samples":   args.NS_maxevals
    }

    # Actually run
    SIMBA.start(args.i, table_url = args.table, comment = "Job started /w %i chains, %i samples, %i burn in and %i cores" %(args.Nchains, nsamples, args.Nburn, args.Ncores))
    output, nest_seeds, nest_full = fit_single_source(banded_data, MCMC_params=MCMC_params, return_nested_full = True, return_nested_seeds=True) #Main MCMC run
    SIMBA.finish(args.i, table_url = args.table, comment = "Job done /w %i chains, %i samples, %i burn in and %i cores" %(args.Nchains, nsamples, args.Nburn, args.Ncores))

    # Outputs
    print("Job finished. Saving to %s" %(job_args["out_url"]+"outchain-%s.dat" %mode) )
    out, out_keys = flatten_dict(output)
    out_nest_seed, out_nest_keys_seed = flatten_dict(nest_seeds)
    out_nest_full, out_nest_keys_full = flatten_dict(nest_full)
    try:
        np.savetxt(job_args["out_url"]+"outchain-%s.dat" %mode",out)
        np.savetxt(job_args["out_url"]+"outchain-nest-seed-%s.dat" %mode",out_nest_seed)
        np.savetxt(job_args["out_url"]+"outchain-nest-full-%s.dat" %mode",out_nest_full)
        np.savetxt(job_args["out_url"]+"outchain_keys-%s.dat" %mode",out_keys,fmt="%s")
    except:
        print("Unable to locate output folder %s. Saving to local runtime folder instead" %job_args["out_url"])
        np.savetxt("./outchain-%s.dat" %mode,out)
        np.savetxt("./outchain-nest-seed-%s.dat" %mode",out_nest_seed)
        np.savetxt("./outchain-nest-full-%s.dat" %mode",out_nest_full)
        np.savetxt("./outchain_keys-%s.dat" %mode",out_keys,fmt="%s")

if __name__=="__main__":
    main()
