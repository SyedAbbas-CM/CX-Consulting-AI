#!/bin/bash

# AWS Transfer Script for CX Consulting AI
# Transfers models, vector stores, and essential data to AWS VM

# Configuration
AWS_KEY="CX-Consulting-AI.pem"
AWS_HOST="ubuntu@ec2-13-60-53-103.eu-north-1.compute.amazonaws.com"
REMOTE_DIR="/home/ubuntu/CX-Consulting-AI"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting transfer to AWS VM...${NC}"

# Function to check if SSH key exists
check_ssh_key() {
    if [ ! -f "$AWS_KEY" ]; then
        echo -e "${RED}❌ SSH key $AWS_KEY not found!${NC}"
        echo -e "${YELLOW}💡 Make sure the key is in the current directory or update the AWS_KEY variable${NC}"
        exit 1
    fi
    chmod 600 "$AWS_KEY"
}

# Function to test SSH connection
test_connection() {
    echo -e "${YELLOW}Testing SSH connection...${NC}"
    if ssh -i "$AWS_KEY" -o ConnectTimeout=10 "$AWS_HOST" "echo 'Connection successful'"; then
        echo -e "${GREEN}✅ SSH connection successful${NC}"
    else
        echo -e "${RED}❌ SSH connection failed${NC}"
        exit 1
    fi
}

# Function to create remote directories
create_remote_dirs() {
    echo -e "${YELLOW}Creating remote directory structure...${NC}"
    ssh -i "$AWS_KEY" "$AWS_HOST" "
        mkdir -p $REMOTE_DIR/{models,data,app/data}
        mkdir -p $REMOTE_DIR/app/data/{vectorstore,projects,uploads,documents,templates,chunked}
    "
}

# Function to transfer files with progress
transfer_with_progress() {
    local source="$1"
    local dest="$2"
    local description="$3"

    echo -e "${BLUE}📁 Transferring $description...${NC}"

    # Use rsync with compression, progress, and resume capability
    rsync -avz --progress --partial --stats \
        -e "ssh -i $AWS_KEY" \
        "$source" "$AWS_HOST:$dest"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $description transferred successfully${NC}"
    else
        echo -e "${RED}❌ Failed to transfer $description${NC}"
        return 1
    fi
}

# Function to get directory size
get_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

# Main transfer function
main_transfer() {
    echo -e "${BLUE}📊 Analyzing transfer requirements...${NC}"

    # Show sizes of what we're about to transfer
    echo -e "${YELLOW}Transfer Summary:${NC}"
    echo -e "  Models directory: $(get_size models/)"
    echo -e "  Vector store: $(get_size app/data/vectorstore/)"
    echo -e "  Projects data: $(get_size app/data/projects/)"
    echo -e "  Templates: $(get_size app/data/templates/)"
    echo -e "  Documents: $(get_size app/data/documents/)"
    echo ""

    # Ask for confirmation
    read -p "Do you want to proceed with the transfer? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Transfer cancelled${NC}"
        exit 0
    fi

    # Start transfers
    echo -e "${BLUE}🚀 Starting file transfers...${NC}"

    # 1. Transfer models (largest files)
    if [ -d "models/" ]; then
        echo -e "${YELLOW}📦 Transferring models (this may take a while for large models)...${NC}"
        transfer_with_progress "models/" "$REMOTE_DIR/" "AI Models"
    fi

    # 2. Transfer vector store data
    if [ -d "app/data/vectorstore/" ]; then
        transfer_with_progress "app/data/vectorstore/" "$REMOTE_DIR/app/data/" "Vector Store"
    fi

    # 3. Transfer projects data
    if [ -d "app/data/projects/" ]; then
        transfer_with_progress "app/data/projects/" "$REMOTE_DIR/app/data/" "Projects Data"
    fi

    # 4. Transfer templates
    if [ -d "app/data/templates/" ]; then
        transfer_with_progress "app/data/templates/" "$REMOTE_DIR/app/data/" "Templates"
    fi

    # 5. Transfer documents
    if [ -d "app/data/documents/" ]; then
        transfer_with_progress "app/data/documents/" "$REMOTE_DIR/app/data/" "Documents"
    fi

    # 6. Transfer other data directories
    if [ -d "app/data/chunked/" ]; then
        transfer_with_progress "app/data/chunked/" "$REMOTE_DIR/app/data/" "Chunked Data"
    fi

    # 7. Transfer users database
    if [ -f "app/data/users.db" ]; then
        transfer_with_progress "app/data/users.db" "$REMOTE_DIR/app/data/" "Users Database"
    fi

    # 8. Transfer any global data
    if [ -d "data/" ]; then
        transfer_with_progress "data/" "$REMOTE_DIR/" "Global Data"
    fi
}

# Function to verify transfer
verify_transfer() {
    echo -e "${YELLOW}🔍 Verifying transfer...${NC}"

    ssh -i "$AWS_KEY" "$AWS_HOST" "
        echo 'Remote directory structure:'
        find $REMOTE_DIR -maxdepth 3 -type d | sort
        echo ''
        echo 'Disk usage:'
        du -sh $REMOTE_DIR/{models,app/data} 2>/dev/null || true
    "
}

# Function to set permissions
set_permissions() {
    echo -e "${YELLOW}🔧 Setting correct permissions...${NC}"

    ssh -i "$AWS_KEY" "$AWS_HOST" "
        chmod -R 755 $REMOTE_DIR/models/
        chmod -R 755 $REMOTE_DIR/app/data/
        chown -R ubuntu:ubuntu $REMOTE_DIR/
    "
}

# Main execution
echo -e "${BLUE}=== CX Consulting AI - AWS Transfer Script ===${NC}"
echo -e "${YELLOW}Target: $AWS_HOST${NC}"
echo -e "${YELLOW}Remote Directory: $REMOTE_DIR${NC}"
echo ""

check_ssh_key
test_connection
create_remote_dirs
main_transfer
set_permissions
verify_transfer

echo -e "${GREEN}🎉 Transfer completed successfully!${NC}"
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "  1. ssh -i $AWS_KEY $AWS_HOST"
echo -e "  2. cd $REMOTE_DIR"
echo -e "  3. source venv/bin/activate"
echo -e "  4. pip install -r requirements.txt"
echo -e "  5. uvicorn app.main:app --host 0.0.0.0 --port 8000"
