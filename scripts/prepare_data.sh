#!/bin/bash
#SBATCH --job-name=prepare-data-cpu
#SBATCH --time=0-8:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --ntasks=1

module load libffi
source ~/anaconda3/bin/activate
conda activate my-rdkit-env
export HF_HOME=$SCRATCH/hf_home

python prepare_data.py --config-name=ZINC_22_smiles_atomwise
