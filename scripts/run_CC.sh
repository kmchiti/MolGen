#!/bin/bash
#SBATCH --job-name=molgpt
#SBATCH --time=0-03:00

#SBATCH --partition=unkillable    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu

module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load git-lfs/3.4.0 rust/1.70.0 protobuf/24.4 arrow/15.0.1
module load rdkit/2023.09.3
source ../ENV/bin/activate
export HF_HOME=$SCRATCH/hf_home

# wandb login --relogin 73fd65ff1623ce64c1f20ed621c065ec55d7eaa3

python train.py --config-name=config_moses
conda deactivate