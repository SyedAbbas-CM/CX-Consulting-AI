#!/bin/bash

# Quick AWS Transfer - Just Gemma 12B + Vector Store
AWS_KEY="CX-Consulting-AI.pem"
AWS_HOST="ubuntu@ec2-13-60-53-103.eu-north-1.compute.amazonaws.com"
REMOTE_DIR="/home/ubuntu/CX-Consulting-AI"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Quick Transfer: Gemma 12B + Vector Store${NC}"

# Check SSH key
if [ ! -f "$AWS_KEY" ]; then
    echo -e "${RED}❌ SSH key $AWS_KEY not found!${NC}"
    echo -e "${YELLOW}💡 Put your CX-Consulting-AI.pem file in this directory${NC}"
    exit 1
fi
chmod 600 "$AWS_KEY"

# Test connection
echo -e "${YELLOW}Testing connection...${NC}"
if ! ssh -i "$AWS_KEY" -o ConnectTimeout=10 "$AWS_HOST" "echo 'Connected!'" 2>/dev/null; then
    echo -e "${RED}❌ Can't connect to AWS. Check your key and instance.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Connected successfully${NC}"

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
ssh -i "$AWS_KEY" "$AWS_HOST" "mkdir -p $REMOTE_DIR/{models,app/data/vectorstore}"

# Transfer Gemma 12B model
echo -e "${BLUE}📦 Transferring Gemma 12B (6.4GB)...${NC}"
rsync -avz --progress --partial \
    -e "ssh -i $AWS_KEY" \
    models/gemma-12B-it.QAT-Q4_0.gguf \
    "$AWS_HOST:$REMOTE_DIR/models/"

# Transfer vector store if it exists
if [ -d "app/data/vectorstore/" ]; then
    echo -e "${BLUE}📁 Transferring Vector Store...${NC}"
    rsync -avz --progress --partial \
        -e "ssh -i $AWS_KEY" \
        app/data/vectorstore/ \
        "$AWS_HOST:$REMOTE_DIR/app/data/vectorstore/"
fi

# Transfer users database if it exists
if [ -f "app/data/users.db" ]; then
    echo -e "${BLUE}👤 Transferring Users Database...${NC}"
    rsync -avz --progress \
        -e "ssh -i $AWS_KEY" \
        app/data/users.db \
        "$AWS_HOST:$REMOTE_DIR/app/data/"
fi

echo -e "${GREEN}🎉 Transfer completed!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  ssh -i $AWS_KEY $AWS_HOST"
echo -e "  cd $REMOTE_DIR && source venv/bin/activate"
echo -e "  pip install -r requirements.txt"
