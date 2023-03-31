#!/bin/bash
#SBATCH --job-name=getafixtestjob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-10:00:00

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

python activate_test.py > _start.txt

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 6

#If finished properly, print the time to a 'done' file
python activate_test.py > _done.txt