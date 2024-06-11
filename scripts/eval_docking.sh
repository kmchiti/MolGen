#!/bin/bash
#SBATCH --job-name=docking-eval
#SBATCH --time=0-03:00

#SBATCH --partition=long-cpu    # ask for unkillable job
#SBATCH --cpus-per-task=8              # number of cpus per gpu
#SBATCH --mem=32G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

python eval_docking.py --preprocess_num_jobs 8 --config_name train_ZINC_270M_atomwise --batch_size 4096 --target fa7
conda deactivate