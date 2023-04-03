#!/bin/bash
#SBATCH --job-name=HM_line_fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0-30:00:00
#SBATCH --array=0-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 6 -Nchains 300 -Nburn 200 -Nsamples 600 -i $SLURM_ARRAY_TASK_ID