#!/bin/bash

# Fix model path issue on AWS
AWS_KEY="CX-Consulting-AI.pem"
AWS_HOST="ubuntu@ec2-13-60-53-103.eu-north-1.compute.amazonaws.com"

echo "🔧 Fixing model path to use absolute path..."

ssh -i "$AWS_KEY" "$AWS_HOST" << 'ENDSSH'
cd /home/ubuntu/CX-Consulting-AI

# Create .env with absolute path
cat > .env << 'EOF'
# Model Configuration for AWS - Using absolute paths
MODEL_PATH=/home/ubuntu/CX-Consulting-AI/models/gemma-12B-it.QAT-Q4_0.gguf
MODEL_ID=google/gemma-12b-it
LLM_BACKEND=llama.cpp
CHAT_FORMAT=gemma
MAX_MODEL_LEN=8192
N_THREADS=4
LLAMA_CPP_VERBOSE=false

# Memory settings
REDIS_URL=redis://localhost:6379/0

# Deployment
DEPLOYMENT_MODE=aws
DEBUG=false
EOF

echo "✅ Updated .env with absolute model path:"
cat .env | grep MODEL_PATH

# Verify the file exists at absolute path
if [ -f "/home/ubuntu/CX-Consulting-AI/models/gemma-12B-it.QAT-Q4_0.gguf" ]; then
    echo "✅ Model file verified at absolute path"
    ls -lh /home/ubuntu/CX-Consulting-AI/models/gemma-12B-it.QAT-Q4_0.gguf
else
    echo "❌ Model file still not found!"
fi

# Also check that we're in the right directory when starting
echo "📍 Current working directory: $(pwd)"
echo "📁 Directory contents:"
ls -la

echo "🚀 Now try starting the server:"
echo "source venv/bin/activate"
echo "uvicorn app.main:app --host 0.0.0.0 --port 8000"

ENDSSH

echo "✅ Model path fix complete!"
