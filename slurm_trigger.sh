#!/bin/bash
#SBATCH --job-name=HM_line_fit-job-%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-10:00:00
#SBATCH --array=2-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com
#SBATCH --output=./slurm_logs/slurm-%j.out

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 2 -Nchains 2 -Nburn 200 -Nsamples 600 -i $SLURM_ARRAY_TASK_ID
