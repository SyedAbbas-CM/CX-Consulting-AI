#!/bin/bash
# AWS GPU VM Deployment Script for CX Consulting AI

set -e

echo "🚀 Starting AWS GPU deployment for CX Consulting AI..."

# Check if we're on an AWS instance
if ! curl -s --max-time 5 http://169.254.169.254/latest/meta-data/instance-id > /dev/null; then
    echo "⚠️  Warning: This doesn't appear to be an AWS instance"
fi

# Check for GPU
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: No NVIDIA GPU detected. Use a GPU-enabled instance (g4dn, g5, p3, etc.)"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake python3-dev python3-venv redis-server

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install CUDA-enabled PyTorch FIRST
echo "🔥 Installing CUDA-enabled PyTorch..."
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install CUDA-enabled llama-cpp-python
echo "🦙 Installing CUDA-enabled llama-cpp-python..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.3.1 --force-reinstall --no-cache-dir

# Install remaining requirements
echo "📚 Installing remaining dependencies..."
pip install -r requirements.txt

# Install any missing packages that might not be in requirements.txt
echo "🔧 Installing additional required packages..."
pip install starlette-prometheus==0.10.0 email-validator==2.2.0 jwt==1.3.1 psutil==5.9.1 faiss-cpu==1.9.0

# Verify critical imports work
echo "🧪 Testing critical imports..."
python -c "
import faiss
import psutil
import jwt
import email_validator
from starlette_prometheus import PrometheusMiddleware
print('✅ All critical packages imported successfully')
"

# Set up environment variables
echo "⚙️  Configuring environment for AWS..."
cat > .env << EOF
# AWS GPU VM Configuration
DEPLOYMENT_MODE=aws
EMBEDDING_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0
FLASH_ATTENTION=true
N_THREADS=$(nproc)
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"
OMP_NUM_THREADS=$(nproc)

# LLM Settings
MODEL_ID=google/gemma-7b-it
MODEL_PATH="models/gemma-7b-it.Q4_K_M.gguf"
LLM_BACKEND=llama.cpp
GPU_COUNT=1
MAX_MODEL_LEN=8192

# Server Settings
HOST=0.0.0.0
PORT=8000
REDIS_URL=redis://localhost:6379/0

# Performance
LLM_TIMEOUT=300
EOF

# Start Redis
echo "🚀 Starting Redis server..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test GPU functionality
echo "🧪 Testing GPU functionality..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Create systemd service (optional)
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/cx-consulting-ai.service > /dev/null << EOF
[Unit]
Description=CX Consulting AI
After=network.target redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin:/usr/bin:/bin
ExecStart=$(pwd)/venv/bin/python start.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo "✅ AWS GPU deployment complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python start.py"
echo ""
echo "📝 Or use systemd:"
echo "   sudo systemctl start cx-consulting-ai"
echo "   sudo systemctl status cx-consulting-ai"
echo ""
echo "🌐 Application will be available at http://your-instance-ip:8000"
