#!/bin/bash
#SBATCH --job-name=molgpt
#SBATCH --time=0-03:00

#SBATCH --partition=main    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6              # number of cpus per gpu
#SBATCH --mem-per-gpu=16G               # memory per gpu
#SBATCH --exclude=cn-k[001-002]

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

# wandb login --relogin 73fd65ff1623ce64c1f20ed621c065ec55d7eaa3

python finetune.py --exp_name=llama_small_FA_ZINC_270M_3d67f607_ --config_name=finetune_ZINC_270M_atomwise --task qed
conda deactivate