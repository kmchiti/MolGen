#!/bin/bash
#SBATCH --job-name=RL
#SBATCH --time=0-03:00

#SBATCH --partition=long    # ask for unkillable job
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --exclude=cn-k[001-002]

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

# wandb login --relogin 73fd65ff1623ce64c1f20ed621c065ec55d7eaa3

#python train.py --config-name=train_moses_gpt2
python My_RL.py --batch_size 120 --target fa7 --num_oracle 40000 --num_proc 24 --wandb

conda deactivate