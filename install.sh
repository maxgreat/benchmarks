# Install Python
sudo apt-get update
sudo apt-get install -y python3 python3-pip
sudo apt-get install -y cmake

# Upgrade pip3 to the latest version
python3 -m pip install --upgrade pip

# Install PyTorch and extensions
python3 -m pip install torch
python3 -m pip install torchvision torchaudio torchtext

# Turn on Graviton3 optimization
export DNNL_DEFAULT_FPMATH_MODE=BF16
export LRU_CACHE_CAPACITY=1024

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release

wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q8_0.gguf?download=true
wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q6_K.gguf?download=true
wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q5_K_M.gguf?download=true
wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf?download=true
