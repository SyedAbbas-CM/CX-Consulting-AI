#!/bin/bash

# CX Consulting AI - Simple Model Downloader
# Downloads models from Google Drive and sets up everything

set -e

echo "🚀 CX Consulting AI - Model Setup"
echo "=================================="

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "📦 Installing gdown..."
    pip install gdown
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p app/data/vectorstore
mkdir -p app/data/projects
mkdir -p app/data/documents

# Download models from Google Drive
echo "⬇️ Downloading models (this may take a while)..."

# Replace these with your actual Google Drive file IDs
# Format: gdown "https://drive.google.com/uc?id=FILE_ID" -O filename

# Gemma 2B (1.5GB) - Fast model for testing
gdown "1your_file_id_here" -O models/gemma-2b-it.Q4_K_M.gguf

# Gemma 7B (5GB) - Balanced model
# gdown "1your_file_id_here" -O models/gemma-7b-it.Q4_K_M.gguf

# Vector store
echo "⬇️ Downloading vector store..."
gdown "1your_vectorstore_id_here" -O vectorstore.zip
unzip -q vectorstore.zip -d app/data/
rm vectorstore.zip

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file..."
    cat > .env << EOF
# Model Configuration
MODEL_PATH=models/gemma-2b-it.Q4_K_M.gguf
MODEL_TYPE=llama
MAX_TOKENS=2048
TEMPERATURE=0.7

# Database
DATABASE_URL=sqlite:///./app/data/users.db

# Vector Store
VECTOR_STORE_PATH=app/data/vectorstore

# API Configuration
SECRET_KEY=$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000
EOF
fi

# Install dependencies if venv doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python start.py"
echo ""
echo "🌐 Then open: http://localhost:8000"
echo "🔐 Login: azureuser / demo123456"
