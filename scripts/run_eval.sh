#!/bin/bash
#SBATCH --job-name=molgpt
#SBATCH --time=0-03:00

#SBATCH --partition=main    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --exclude=cn-k[001-002]

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

# wandb login --relogin 73fd65ff1623ce64c1f20ed621c065ec55d7eaa3

#python eval.py --config_name=train_moses_gpt2 --evaluate_on_checkpoints --seeds 42 1335 2024
python eval.py --config_name=train_ZINC_270M_atomwise --evaluate_on_checkpoints --seeds 42 --tdc
conda deactivate