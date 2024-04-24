#!/bin/bash
set -e
set -v

export FLASH_ATTN_VERSION='2.3.2'
export DeepSpeed_VERSION='0.11.1'
export MAX_JOBS=4

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
pip install 'PyTDC'
pip install 'rdkit'
pip install 'CIRpy==1.0.2'
pip install 'tabulate==0.9.0'
pip install 'PyTDC==0.3.9'

# Clone and install flash-attention v2
NV_CC="8.0;8.6" # flash-attention-v2 and exllama_kernels are anyway limited to CC of 8.0+
FLASH_ATTENTION_DIR="./flash-attention-v2"
git clone https://github.com/Dao-AILab/flash-attention "$FLASH_ATTENTION_DIR"
pushd "$FLASH_ATTENTION_DIR"
git checkout "tags/v$FLASH_ATTN_VERSION"
TORCH_CUDA_ARCH_LIST="$NV_CC" MAX_JOBS="$MAX_JOBS" python setup.py install
pushd csrc/fused_dense_lib && pip install .
pushd ../xentropy && pip install .
pushd ../rotary && pip install .
pushd ../layer_norm && pip install .
popd  # Exit from csrc/rotary
popd  # Exit from flash-attention

# Clone and install DeepSpeed
DeepSpeed_DIR="./deep_speed"
git clone https://github.com/microsoft/DeepSpeed/ "$DeepSpeed_DIR"
cd "$DeepSpeed_DIR"
git checkout "tags/v$DeepSpeed_VERSION"
rm -rf build
TORCH_CUDA_ARCH_LIST="$NV_CC" DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LION=1 DS_BUILD_QUANTIZER=1 \
pip install . --global-option="build_ext" --global-option="-j4" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
