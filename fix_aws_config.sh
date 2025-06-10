#!/bin/bash

# Script to fix AWS configuration
AWS_KEY="CX-Consulting-AI.pem"
AWS_HOST="ubuntu@ec2-13-60-53-103.eu-north-1.compute.amazonaws.com"
REMOTE_DIR="/home/ubuntu/CX-Consulting-AI"

echo "🔧 Fixing AWS model configuration..."

# SSH into AWS and update the config
ssh -i "$AWS_KEY" "$AWS_HOST" << 'ENDSSH'
cd /home/ubuntu/CX-Consulting-AI

# Create a .env file to override the model settings
cat > .env << 'EOF'
# Model Configuration for AWS
MODEL_PATH=models/gemma-12B-it.QAT-Q4_0.gguf
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

echo "✅ Created .env file with correct model settings"

# Verify the model file exists
if [ -f "models/gemma-12B-it.QAT-Q4_0.gguf" ]; then
    echo "✅ Model file found: $(ls -lh models/gemma-12B-it.QAT-Q4_0.gguf)"
else
    echo "❌ Model file not found!"
    ls -la models/
fi

# Install any missing packages
source venv/bin/activate
pip install werkzeug==3.1.3 PyJWT==2.8.0

echo "🚀 Configuration updated! Try starting the server now:"
echo "uvicorn app.main:app --host 0.0.0.0 --port 8000"

ENDSSH

echo "✅ AWS configuration complete!"
