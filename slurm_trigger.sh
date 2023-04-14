#!/bin/bash
#SBATCH --job-name=HM_line_fit-job-%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=2-00:00:00
#SBATCH --array=20,22,32,40,41,44,45,50,53,55,56,59,62,71,73,74,76,78,79,81,86,88,92,
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com
#SBATCH --output=./slurm_logs/slurm-%j.out

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 1 -Nchain 300 -Nburn 200 -Nsamples 200 -i $SLURM_ARRAY_TASK_ID -progress_bar 1
