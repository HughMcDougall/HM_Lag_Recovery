#!/bin/bash
#SBATCH --job-name=getafixtestjob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-01:00:00
#SBATCH --array=1-1

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate nestconda_dev

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py