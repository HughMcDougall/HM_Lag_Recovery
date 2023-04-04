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
from data_utils import lc_to_banded, flatten_dict


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

    ap.add_argument('-Ncores', '--Ncores',     metavar='Ncores',  type=int,       help='Number of devices to feed to jax',         default=1)
    ap.add_argument('-i', '--i',                 metavar='i',       type=int,       help='job itteration number',                    default=0)
    ap.add_argument('-Nchains', '--Nchains',       metavar='Nchains',  type=int,       help='Number of MCMC chains',                    default=300)
    ap.add_argument('-Nburn', '--Nburn',         metavar='Nburn',   type=int,       help='Number of burn-in steps',                  default=200)
    ap.add_argument('-Nsamples', '--Nsamples',     metavar='Nsamples', type=int,       help='Number of samples',                        default=600)
    ap.add_argument('-step_size', '--step_size', metavar='step_size', type=float,   help='Step Size in HMC',                         default=1E-2)
    ap.add_argument('-progress_bar', '--progress_bar', metavar='progress_bar', type=float,   help='numpyro progress bar',                         default=False)
    ap.add_argument('-table', '--table', metavar='table', type=str,   help='SIMBA table to draw jobs from',                         default=SIMBA._def_tab_url)
    
    args = ap.parse_args()

    #=======================
    #GET AND NORMALIZE DATA

    #Get location of sources
    job_args = SIMBA.get_args(args.i, table_url=args.table)

    print("Beginning job %i on job list %s)" %(args.i, args.table))

    #Read files and sort into banded form.
    #Normalize and shift data in this runtime
    lcs_unbanded = []
    for url in [job_args["cont_url"], job_args["line1_url"], job_args["line2_url"]]:
        data=np.loadtxt(url)
        
        w = data[:,2]**-2
        wsum = np.sum(w)

        mean = np.sum(data[:,1]* w) / wsum)
        std  = np.sqrt( np.sum((data[:,1]-mean)**2 * w) / wsum)
        
        lcs_unbanded.append({
            "T": data[:,0],
            "Y": ( data[:,1]-mean ) / std,
            "E": data[:,2] / std ,
        })

    banded_data = lc_to_banded(lcs_unbanded)
    banded_data['T']-=np.min(banded_data['T'])

    #Save data output to be safe
    out, out_keys = flatten_dict(banded_data)
    print("Data loaded. Saving normalized lightcurve to %s" %(job_args["out_url"]+"banded_data.dat"))
    try:
        np.savetxt(job_args["out_url"]+"banded_data.dat",out)
    except:
        print("Unable to locate output folder %s. Saving to local runtime folder instead" %job_args["out_url"])
        np.savetxt("./banded_data.dat",out)
        
    #=======================
    #PERFORM FITTING

    MCMC_params ={
        "Ncores":       args.Ncores,
        "Nchain":       args.Nchains,
        "Nburn":        args.Nburn,
        "Nsample":      args.Nsamples,
        "step_size":    args.step_size,
        "progress_bar": args.progress_bar
    }


    SIMBA.start(args.i, table_url = args.table, comment = "Job started /w %i chains, %i samples, %i burn in and %i cores" %(args.Nchains, args.Nsamples, args.Nburn, args.Ncores))
    output = fit_single_source(banded_data, params=MCMC_params) #Main MCMC run
    SIMBA.finish(args.i, table_url = args.table, comment = "Job done /w %i chains, %i samples, %i burn in and %i cores" %(args.Nchains, args.Nsamples, args.Nburn, args.Ncores))

    #Outputs
    print("Job finished. Saving to %s" %(job_args["out_url"]+"outchain.dat") )
    out,out_keys = flatten_dict(output)
    try:
        np.savetxt(job_args["out_url"]+"outchain.dat",out)
        np.savetxt(job_args["out_url"]+"outchain_keys.dat",out_keys,fmt="%s")
    except:
        print("Unable to locate output folder %s. Saving to local runtime folder instead" %job_args["out_url"])
        np.savetxt("./outchain.dat",out)
        np.savetxt("./outchain_keys.dat",out_keys,fmt="%s")

if __name__=="__main__":
    main()
