# Install Python
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Upgrade pip3 to the latest version
python3 -m pip install --upgrade pip

# Install PyTorch and extensions
python3 -m pip install torch
python3 -m pip install torchvision torchaudio torchtext

# Turn on Graviton3 optimization
export DNNL_DEFAULT_FPMATH_MODE=BF16
export LRU_CACHE_CAPACITY=1024

export THP_MEM_ALLOC_ENABLE=1


