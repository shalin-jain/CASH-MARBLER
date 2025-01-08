#!/bin/bash
#SBATCH --job-name=sjain441-cash-marbler
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=V100:1
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

#####################
# FOR USE ON PACE #
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