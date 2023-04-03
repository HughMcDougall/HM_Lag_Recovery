#!/bin/bash
#SBATCH --job-name=LineFit01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=500M
#SBATCH --time=0-00:10:00
#SBATCH --array=0-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

python activate_test.py > _start.txt

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 1 -Nchains 1 -Nburn 0 -Nsamples 1 -i $SLURM_ARRAY_TASK_ID

#If finished properly, print the time to a 'done' file
python activate_test.py > _done.txt