#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --time=3-00:00

#SBATCH --partition=lab-chandar
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gpus-per-task=4               # number of gpus per node
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

torchrun --nproc_per_node=4 train.py --config-name=config_PubChem_atomwise
