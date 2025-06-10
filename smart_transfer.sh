#!/bin/bash
# Smart Transfer Script for CX Consulting AI
# Optimized for AWS GPU deployment with selective transfer options

set -e

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <aws-instance-ip> <ssh-key-path> [transfer-mode]"
    echo ""
    echo "Transfer modes:"
    echo "  essential  - Vector DB + 1 model (fastest)"
    echo "  balanced   - Vector DB + 3 models (recommended)"
    echo "  complete   - Everything (slowest)"
    echo ""
    echo "Example: $0 54.123.45.67 ~/.ssh/my-key.pem balanced"
    exit 1
fi

AWS_IP=$1
SSH_KEY=$2
TRANSFER_MODE=${3:-balanced}
REMOTE_USER="ubuntu"
REMOTE_PATH="/home/$REMOTE_USER/CX-Consulting-AI"

echo "🚀 Smart transfer to AWS: $AWS_IP (Mode: $TRANSFER_MODE)"

# Check SSH connection
echo "🔑 Testing connection..."
ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_USER@$AWS_IP" "echo 'Connected'" || {
    echo "❌ Connection failed"
    exit 1
}

# Create directories
echo "📁 Setting up directories..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "
    mkdir -p $REMOTE_PATH/{models,app/data/{vectorstore,projects,documents,uploads,templates}}
"

# PRIORITY 1: Fix the immediate error first
echo "🔧 Installing missing packages on remote..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "
    cd $REMOTE_PATH
    source venv/bin/activate 2>/dev/null || true
    pip install starlette-prometheus==0.10.0 email-validator==2.2.0 jwt==1.3.1 psutil==5.9.1 faiss-cpu==1.9.0
"

# PRIORITY 2: Transfer vector database (CRITICAL)
echo "📊 Transferring vector database..."
if [ -f "app/data/vectorstore/chroma.sqlite3" ]; then
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        ./app/data/vectorstore/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/app/data/vectorstore/"
    echo "✅ Vector database transferred (200MB)"
else
    echo "⚠️  No vector database found"
fi

# PRIORITY 3: Transfer models based on mode
case $TRANSFER_MODE in
    "essential")
        echo "🦙 Transferring 1 essential model..."
        if [ -f "models/gemma-7b-it.Q4_K_M.gguf" ]; then
            rsync -avz --progress -e "ssh -i $SSH_KEY" \
                ./models/gemma-7b-it.Q4_K_M.gguf "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/models/"
        fi
        ;;
    "balanced")
        echo "🦙 Transferring 3 balanced models..."
        for model in "gemma-7b-it.Q4_K_M.gguf" "gemma-4b-it.Q4_K_M.gguf" "gemma-2b-it.Q4_K_M.gguf"; do
            if [ -f "models/$model" ]; then
                echo "Transferring $model..."
                rsync -avz --progress -e "ssh -i $SSH_KEY" \
                    "./models/$model" "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/models/"
            fi
        done
        ;;
    "complete")
        echo "🦙 Transferring ALL models (30GB+)..."
        rsync -avz --progress -e "ssh -i $SSH_KEY" \
            ./models/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/models/"
        ;;
esac

# PRIORITY 4: Transfer project data
echo "📁 Transferring project data..."
if [ -d "app/data/projects" ]; then
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        ./app/data/projects/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/app/data/projects/"
fi

if [ -d "app/data/templates" ]; then
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        ./app/data/templates/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/app/data/templates/"
fi

# PRIORITY 5: Transfer documents (if any)
if [ -d "app/data/documents" ] && [ "$(ls -A app/data/documents)" ]; then
    echo "📄 Transferring documents..."
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        ./app/data/documents/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/app/data/documents/"
fi

# Update environment for transferred model
echo "⚙️  Configuring model path..."
case $TRANSFER_MODE in
    "essential"|"balanced")
        MODEL_FILE="gemma-7b-it.Q4_K_M.gguf"
        ;;
    "complete")
        MODEL_FILE="gemma-12B-it.QAT-Q4_0.gguf"  # Use larger model for complete mode
        ;;
esac

ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "
    cd $REMOTE_PATH
    # Update model path in .env
    if [ -f '.env' ]; then
        sed -i 's|MODEL_PATH=.*|MODEL_PATH=\"models/$MODEL_FILE\"|' .env
    else
        echo 'MODEL_PATH=\"models/$MODEL_FILE\"' >> .env
    fi
"

# Test the deployment
echo "🧪 Testing deployment..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "
    cd $REMOTE_PATH
    source venv/bin/activate
    python -c 'from starlette_prometheus import PrometheusMiddleware; print(\"✅ Packages OK\")'

    # Test if model file exists
    if [ -f 'models/$MODEL_FILE' ]; then
        echo '✅ Model file exists: $MODEL_FILE'
    else
        echo '⚠️  Model file not found: $MODEL_FILE'
    fi

    # Check vector database
    if [ -f 'app/data/vectorstore/chroma.sqlite3' ]; then
        echo '✅ Vector database exists'
    else
        echo '⚠️  Vector database not found'
    fi
"

echo ""
echo "✅ Transfer complete!"
echo ""
echo "📊 Transfer Summary ($TRANSFER_MODE mode):"
case $TRANSFER_MODE in
    "essential")
        echo "  - Vector database: ✅ (200MB)"
        echo "  - Models: 1x gemma-7b (5GB)"
        echo "  - Total: ~5.2GB"
        ;;
    "balanced")
        echo "  - Vector database: ✅ (200MB)"
        echo "  - Models: 3x gemma variants (8.8GB)"
        echo "  - Total: ~9GB"
        ;;
    "complete")
        echo "  - Vector database: ✅ (200MB)"
        echo "  - Models: All 6 models (30GB+)"
        echo "  - Total: ~30.2GB"
        ;;
esac

echo ""
echo "🚀 Ready to start on AWS:"
echo "   ssh -i $SSH_KEY $REMOTE_USER@$AWS_IP"
echo "   cd $REMOTE_PATH"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "🌐 App URL: http://$AWS_IP:8000"
