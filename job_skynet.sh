#!/bin/bash
#SBATCH --job-name=sjain441-cash-marbler
#SBATCH --partition=ravichandar-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --exclude=heistotron

#####################
# FOR USE ON SKYNET #
#####################

# Wiki said to put this here
export PYTHONUNBUFFERED=TRUE

# source conda
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate

# source our conda env
conda activate MARBLER

# run our train command via SLURM
srun -u python -u ~/CASH-MARBLER/epymarl/src/main.py --config=qmix --env-config=gymma