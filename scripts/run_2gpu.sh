#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --time=0-12:00

#SBATCH --partition=main    # ask for unkillable job
#SBATCH --gpus-per-task=2               # number of gpus per node
#SBATCH --cpus-per-task=8              # number of cpus per gpu
#SBATCH --mem-per-gpu=16G               # memory per gpu
#SBATCH --ntasks=1
#SBATCH --constraint=80gb               # constraints

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

torchrun --nproc_per_node=2 train.py --config-name=config_moses
