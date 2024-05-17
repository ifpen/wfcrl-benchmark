#!/bin/bash

#SBATCH --job-name=ippo_wind_rose_Turb3      # Job name
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1                    # A single task (process)
#SBATCH --cpus-per-task=1             # Using a single CPU
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx6000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=100                # Total memory allocated
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x_%A_%a.out         # Standard output and error log. %x denotes the job name, %A the jobid, %a the taskid.


# job Submitted batch job 2308790

module purge
module load cuda/11.4.0
module load gnu8 openmpi3

echo "Running job on a single cpu"

# Load your environment (conda, ...)
source /home/$USER/.bashrc
conda activate wfcrl_algos

python algos/baseline_ippo_windrose.py --seed ${SLURM_ARRAY_TASK_ID} --env_id Dec_Turb3_Row1_Floris --total_timesteps 50000 --freq_eval 5