module load gcc/9.3.0
module load python/3.10
module load cuda/11.8

python -m venv ../ENV
source ../ENV/bin/activate
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install 'datasets==2.14.5'
pip install 'transformers==4.34'
pip install 'wandb==0.15.12'
pip install 'matplotlib==3.7.0'
pip install 'accelerate==0.23.0'
pip install 'hydra-core==1.3'
pip install 'gdown==4.7.1'
# install flash-attention v2.3.2 from compiled file
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

