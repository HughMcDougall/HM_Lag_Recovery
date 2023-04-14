#!/bin/bash
#SBATCH --job-name=HM_line_fit-job-%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=4,5,6,7,10,11,13,14,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,44,45,46,48,49,50,52,53,55,56,59,62,63,64,65,70,71,72,73,74,76,77,78,79,81,82,84,85,86,88,89,92,
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com
#SBATCH --output=./slurm_logs/slurm-%j.out

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 1 -Nchain 300 -Nburn 200 -Nsamples 200 -i $SLURM_ARRAY_TASK_ID -progress_bar 1