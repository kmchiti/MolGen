#!/bin/bash
#SBATCH --job-name=molgpt
#SBATCH --time=0-03:00

#SBATCH --partition=short-unkillable    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu

module load libffi
source ../ENV/bin/activate
export HF_HOME=$SCRATCH/hf_home

# wandb login --relogin 73fd65ff1623ce64c1f20ed621c065ec55d7eaa3

python train.py
