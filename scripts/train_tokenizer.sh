#!/bin/bash
#SBATCH --job-name=llama-cpu
#SBATCH --time=0-3:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --ntasks=1

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

python dataset/tokenizer_train.py --dataset_name "PubChem" --mol_type "SMILES" --tokenizer_method "tokenizer_method" --max_vocab_size 2048
