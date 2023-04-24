#!/bin/bash
#SBATCH --job-name=HM_line_fit-job-%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=2-00:00:00
#SBATCH --array=101,127,139,148,153,154,155,161,166,172,173 174,175,176,178,180,182,183
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com
#SBATCH --output=./slurm_logs/slurm-%j.out

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python LineFit02.py -Ncores 1 -Nchain 300 -Nburn 1000 -Nsamples 200 -i $SLURM_ARRAY_TASK_ID -progress_bar 0
