#!/bin/bash

# Install Miniconda if not already installed
if [ ! -d ~/miniconda3 ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Create and activate conda environment with Python 3.9
conda create -y -n ml_env python=3.9
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_env

# Install PyTorch with CUDA support using pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip3 install -r requirements.txt

# Verify PyTorch CUDA availability
echo "Checking PyTorch CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA device count:', torch.cuda.device_count())"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available')" 