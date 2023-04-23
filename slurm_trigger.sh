#!/bin/bash
#SBATCH --job-name=HM_line_fit-job-%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=4,11,12,25,33,38,51,54,55,57,59,60,61,63,65,67,69,71,76,77,79,81,90,94,95,98,100,101,102,103,104,105,106,107,109,110,111,116,119,121,123,126,128,129,130,131,132,133,134,135,136,137,138,140,141,142,143,145,147,150,151,152,157,158,159,169,
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com
#SBATCH --output=./slurm_logs/slurm-%j.out

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit02.py -Ncores 1 -Nchain 300 -Nburn 1000 -Nsamples 200 -i $SLURM_ARRAY_TASK_ID -progress_bar 0