#!/bin/bash
# Transfer Models and Data to AWS VM
# Usage: ./transfer_to_aws.sh <aws-instance-ip> <key-file-path>

set -e

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <aws-instance-ip> <ssh-key-path>"
    echo "Example: $0 54.123.45.67 ~/.ssh/my-key.pem"
    exit 1
fi

AWS_IP=$1
SSH_KEY=$2
REMOTE_USER="ubuntu"
REMOTE_PATH="/home/$REMOTE_USER/CX-Consulting-AI"

echo "🚀 Starting transfer to AWS instance: $AWS_IP"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    exit 1
fi

# Set proper permissions on SSH key
chmod 600 "$SSH_KEY"

# Test SSH connection
echo "🔑 Testing SSH connection..."
ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$REMOTE_USER@$AWS_IP" "echo 'SSH connection successful'" || {
    echo "❌ SSH connection failed. Check your IP and key."
    exit 1
}

# Create directories on remote
echo "📁 Creating directories on remote instance..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "
    mkdir -p $REMOTE_PATH/app/data/{models,vectorstore,embeddings,documents,uploads,projects,templates}
    mkdir -p $REMOTE_PATH/models
"

# Transfer models (if they exist)
if [ -d "models" ]; then
    echo "🦙 Transferring LLM models..."
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        ./models/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/models/"
else
    echo "⚠️  No models directory found. You'll need to download models on the AWS instance."
fi

# Transfer app/data directory
if [ -d "app/data" ]; then
    echo "📊 Transferring embeddings and vector database..."
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        --exclude='*.log' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        ./app/data/ "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/app/data/"
else
    echo "⚠️  No app/data directory found."
fi

# Transfer any existing .env file (optional)
if [ -f ".env" ]; then
    echo "⚙️  Transferring environment configuration..."
    scp -i "$SSH_KEY" .env "$REMOTE_USER@$AWS_IP:$REMOTE_PATH/.env.backup"
    echo "📝 Backed up your .env as .env.backup on remote"
fi

# Create a script to download missing models on AWS
echo "📝 Creating model download script on remote..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "cat > $REMOTE_PATH/download_models.sh << 'EOF'
#!/bin/bash
# Download models on AWS instance

echo '🦙 Downloading Gemma 7B model...'
cd $REMOTE_PATH
mkdir -p models

# Download Gemma 7B GGUF model (adjust URL as needed)
# You'll need to get the actual download link for your preferred model
echo 'Please manually download your GGUF model to the models/ directory'
echo 'Common sources:'
echo '  - Hugging Face: https://huggingface.co/models'
echo '  - TheBloke quantized models'
echo '  - Direct GGUF files'

echo '📥 Example download commands:'
echo 'wget -O models/gemma-7b-it.Q4_K_M.gguf [YOUR_MODEL_URL]'
echo 'or'
echo 'huggingface-cli download microsoft/DialoGPT-medium --local-dir models/gemma-7b-it'

# Set executable
chmod +x $REMOTE_PATH/download_models.sh
EOF"

# Create Redis data backup/restore helper
echo "💾 Creating Redis backup script..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "cat > $REMOTE_PATH/setup_redis_data.sh << 'EOF'
#!/bin/bash
# Setup Redis data on AWS

echo '🔄 Starting Redis...'
sudo systemctl start redis-server
sudo systemctl enable redis-server

echo '📊 Redis is ready for your application data'
echo 'Your vector embeddings and project data will be rebuilt when you first run the app'

# Optionally restore Redis data if you have a dump
if [ -f 'redis_dump.rdb' ]; then
    echo '📥 Restoring Redis data...'
    sudo systemctl stop redis-server
    sudo cp redis_dump.rdb /var/lib/redis/dump.rdb
    sudo chown redis:redis /var/lib/redis/dump.rdb
    sudo systemctl start redis-server
    echo '✅ Redis data restored'
fi
EOF"

# Set permissions on remote scripts
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "
    chmod +x $REMOTE_PATH/download_models.sh
    chmod +x $REMOTE_PATH/setup_redis_data.sh
    chmod +x $REMOTE_PATH/deploy_aws.sh
"

# Check disk space on remote
echo "💽 Checking disk space on remote instance..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$AWS_IP" "df -h | grep -E '(Filesystem|/dev/)'"

echo ""
echo "✅ Transfer complete!"
echo ""
echo "🚀 Next steps on your AWS instance:"
echo "1. SSH into your instance:"
echo "   ssh -i $SSH_KEY $REMOTE_USER@$AWS_IP"
echo ""
echo "2. Run the deployment script:"
echo "   cd $REMOTE_PATH"
echo "   ./deploy_aws.sh"
echo ""
echo "3. Download models (if not transferred):"
echo "   ./download_models.sh"
echo ""
echo "4. Setup Redis data:"
echo "   ./setup_redis_data.sh"
echo ""
echo "5. Install missing dependency and start the app:"
echo "   source venv/bin/activate"
echo "   pip install starlette-prometheus==0.10.0"
echo "   python start.py"
echo ""
echo "🌐 Your app will be available at http://$AWS_IP:8000"
