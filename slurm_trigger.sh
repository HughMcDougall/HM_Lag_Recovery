#!/bin/bash
#SBATCH --job-name=HM_line_fit-job-%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=512G
#SBATCH --time=2-00:00:00
#SBATCH --array=17,18,19,20,21,22,23,24,25,26,27,28,29
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com
#SBATCH --output=./slurm_logs/slurm-%j.out

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit01.py -Ncores 1 -Nchain 600 -Nburn 1000 -Nsamples 200 -i $SLURM_ARRAY_TASK_ID -progress_bar 0 -table SIMBAtest_status.dat
