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
source /srv/rail-lab/flash5/$USER/miniconda3/etc/profile.d/conda.sh
conda deactivate

# source our conda env
conda activate CASH_MARBLER

# run our train command via SLURM
srun -u python -u CASH-MARBLER/epymarl/src/main.py -m --config=qmix --env-config=gymma with env_args.time_limit=1000 env_args.key='robotarium_gym:PredatorCapturePrey-v0'