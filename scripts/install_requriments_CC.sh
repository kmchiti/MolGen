#!/bin/bash
set -e
set -v

export FLASH_ATTN_VERSION='2.3.2'
export DeepSpeed_VERSION='0.11.1'
export MAX_JOBS=4

module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module laod intel/2023.2.1
module load cuda/11.8
module load git-lfs/3.4.0 rust/1.70.0 protobuf/24.4 arrow/15.0.1
module load rdkit/2023.09.3

pip uninstall -y ninja && pip install ninja
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install 'datasets==2.14.5'
pip install 'transformers==4.34'
pip install 'wandb==0.15.12'
pip install 'matplotlib==3.7.0'
pip install 'accelerate==0.27.0'
pip install 'hydra-core==1.3'
pip install 'gdown==4.7.1'
pip install wheel setuptools py-cpuinfo
pip install 'molvs==0.1.1'
pip install 'selfies==2.1.1'
pip install 'tabulate==0.9.0'