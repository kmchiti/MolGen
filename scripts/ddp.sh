#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --time=0-03:00

#SBATCH --partition=short-unkillable    # ask for unkillable job
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gpus-per-task=4               # number of gpus per node
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --constraint=80gb               # constraints

module load libffi
source ../ENV/bin/activate
export HF_HOME=$SCRATCH/hf_home

torchrun --nproc_per_node=4 train.py
