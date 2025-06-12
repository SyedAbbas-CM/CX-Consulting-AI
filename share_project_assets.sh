#!/bin/bash

# CX Consulting AI - Project Asset Sharing Script
# This script helps you share LLM models and vector store with others

set -e

echo "📦 CX Consulting AI - Asset Sharing Helper"
echo "=========================================="
echo ""

# Configuration
PROJECT_NAME="CX-Consulting-AI"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check what assets exist
check_assets() {
    print_header "Checking Available Assets"

    echo "📁 Project structure:"

    # Check models directory
    if [ -d "models/" ]; then
        echo "  📂 models/"
        for model in models/*.gguf; do
            if [ -f "$model" ]; then
                size=$(ls -lh "$model" | awk '{print $5}')
                echo "    📄 $(basename "$model") ($size)"
            fi
        done
    else
        print_warning "No models directory found"
    fi

    # Check vector store
    if [ -d "app/data/vectorstore/" ]; then
        echo "  📂 app/data/vectorstore/"
        if [ "$(ls -A app/data/vectorstore/)" ]; then
            du -sh app/data/vectorstore/* 2>/dev/null | while read size path; do
                echo "    📄 $(basename "$path") ($size)"
            done
        else
            print_warning "Vector store directory is empty"
        fi
    else
        print_warning "No vector store directory found"
    fi

    # Check other important data
    if [ -d "app/data/" ]; then
        echo "  📂 app/data/"
        for item in app/data/*; do
            if [ -d "$item" ] && [ "$(basename "$item")" != "vectorstore" ]; then
                echo "    📁 $(basename "$item")/"
            elif [ -f "$item" ]; then
                size=$(ls -lh "$item" | awk '{print $5}')
                echo "    📄 $(basename "$item") ($size)"
            fi
        done
    fi

    echo ""
}

# Function to create a sharing package
create_sharing_package() {
    print_header "Creating Sharing Package"

    PACKAGE_DIR="shared_assets_${TIMESTAMP}"
    mkdir -p "$PACKAGE_DIR"

    echo "📦 Creating package in: $PACKAGE_DIR"

    # Copy models if they exist
    if [ -d "models/" ] && [ "$(ls -A models/)" ]; then
        print_success "Copying models..."
        cp -r models/ "$PACKAGE_DIR/"
    fi

    # Copy vector store if it exists
    if [ -d "app/data/vectorstore/" ] && [ "$(ls -A app/data/vectorstore/)" ]; then
        print_success "Copying vector store..."
        mkdir -p "$PACKAGE_DIR/app/data/"
        cp -r app/data/vectorstore/ "$PACKAGE_DIR/app/data/"
    fi

    # Copy other important data
    if [ -d "app/data/templates/" ]; then
        print_success "Copying templates..."
        mkdir -p "$PACKAGE_DIR/app/data/"
        cp -r app/data/templates/ "$PACKAGE_DIR/app/data/"
    fi

    if [ -f "app/data/users.db" ]; then
        print_success "Copying user database..."
        mkdir -p "$PACKAGE_DIR/app/data/"
        cp app/data/users.db "$PACKAGE_DIR/app/data/"
    fi

    # Create setup instructions
    cat > "$PACKAGE_DIR/SETUP_INSTRUCTIONS.md" << 'EOF'
# CX Consulting AI - Asset Setup Instructions

## What's included:
- Pre-trained LLM models (GGUF format)
- Vector store with embedded documents
- Template files
- User database (optional)

## Setup Steps:

### 1. Clone the main repository:
```bash
git clone <your-repo-url>
cd CX-Consulting-AI
```

### 2. Copy the shared assets:
```bash
# Copy models
cp -r shared_assets_*/models/* ./models/

# Copy vector store and data
cp -r shared_assets_*/app/data/* ./app/data/

# Make sure directories exist
mkdir -p app/data/vectorstore
mkdir -p app/data/templates
mkdir -p app/data/projects
mkdir -p app/data/documents
```

### 3. Install dependencies:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Configure environment:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings:
# - Model paths
# - API keys
# - Database settings
```

### 5. Start the application:
```bash
# Start backend
python start.py

# In another terminal, start frontend (if needed)
cd app/frontend/cx-consulting-ai-3
npm install
npm run dev
```

## Model Information:
- Models are in GGUF format for llama.cpp
- Default model path should be: `models/your-model.gguf`
- Update MODEL_PATH in .env if needed

## Vector Store:
- Contains pre-embedded documents
- No need to re-embed unless you add new documents
- Compatible with ChromaDB

## Troubleshooting:
1. Make sure all file paths in .env are correct
2. Ensure you have enough RAM for the model size
3. Check that Redis is running for the application
4. Verify all dependencies are installed

## Support:
- Check the main README.md for detailed setup
- Review logs in app.log for errors
- Ensure all file permissions are correct
EOF

    # Create a manifest file
    cat > "$PACKAGE_DIR/MANIFEST.txt" << EOF
CX Consulting AI - Shared Assets Package
Created: $(date)
Package: $PACKAGE_DIR

Contents:
EOF

    find "$PACKAGE_DIR" -type f | while read file; do
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  $file ($size)" >> "$PACKAGE_DIR/MANIFEST.txt"
    done

    # Calculate total size
    total_size=$(du -sh "$PACKAGE_DIR" | awk '{print $1}')
    echo "Total package size: $total_size" >> "$PACKAGE_DIR/MANIFEST.txt"

    print_success "Package created: $PACKAGE_DIR (Size: $total_size)"
    echo ""
}

# Function to create a cloud upload script
create_upload_script() {
    print_header "Creating Upload Scripts"

    # Create Google Drive upload script
    cat > "upload_to_gdrive.sh" << 'EOF'
#!/bin/bash
# Upload to Google Drive using rclone

PACKAGE_DIR="$1"
if [ -z "$PACKAGE_DIR" ]; then
    echo "Usage: $0 <package_directory>"
    exit 1
fi

echo "📤 Uploading $PACKAGE_DIR to Google Drive..."

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Configure rclone for Google Drive (interactive)
echo "Setting up Google Drive connection..."
rclone config

# Upload the package
echo "Uploading package..."
rclone copy "$PACKAGE_DIR" gdrive:CX-Consulting-AI-Shared/ --progress

echo "✅ Upload complete!"
echo "🔗 Share the Google Drive folder with your collaborator"
EOF

    # Create AWS S3 upload script
    cat > "upload_to_s3.sh" << 'EOF'
#!/bin/bash
# Upload to AWS S3

PACKAGE_DIR="$1"
BUCKET_NAME="cx-consulting-ai-shared"

if [ -z "$PACKAGE_DIR" ]; then
    echo "Usage: $0 <package_directory>"
    exit 1
fi

echo "📤 Uploading $PACKAGE_DIR to AWS S3..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Please install AWS CLI first:"
    echo "pip install awscli"
    exit 1
fi

# Create bucket if it doesn't exist
aws s3 mb s3://$BUCKET_NAME 2>/dev/null || true

# Upload the package
aws s3 sync "$PACKAGE_DIR" s3://$BUCKET_NAME/$(basename "$PACKAGE_DIR")/ --delete

# Generate presigned URLs for sharing (valid for 7 days)
echo ""
echo "🔗 Shareable links (valid for 7 days):"
aws s3 ls s3://$BUCKET_NAME/$(basename "$PACKAGE_DIR")/ --recursive | while read line; do
    file=$(echo $line | awk '{print $4}')
    url=$(aws s3 presign s3://$BUCKET_NAME/$file --expires-in 604800)
    echo "  $(basename $file): $url"
done

echo "✅ Upload complete!"
EOF

    # Create torrent creation script
    cat > "create_torrent.sh" << 'EOF'
#!/bin/bash
# Create BitTorrent file for P2P sharing

PACKAGE_DIR="$1"
if [ -z "$PACKAGE_DIR" ]; then
    echo "Usage: $0 <package_directory>"
    exit 1
fi

echo "🌐 Creating torrent for $PACKAGE_DIR..."

# Check if transmission-create is available
if ! command -v transmission-create &> /dev/null; then
    echo "Installing transmission-cli..."
    # For macOS
    if command -v brew &> /dev/null; then
        brew install transmission-cli
    # For Ubuntu/Debian
    elif command -v apt &> /dev/null; then
        sudo apt install transmission-cli
    else
        echo "Please install transmission-cli manually"
        exit 1
    fi
fi

# Create torrent file
TORRENT_FILE="${PACKAGE_DIR}.torrent"
transmission-create -o "$TORRENT_FILE" "$PACKAGE_DIR"

echo "✅ Torrent created: $TORRENT_FILE"
echo "📤 Share this .torrent file with your collaborator"
echo "💡 They can download it using any BitTorrent client"
EOF

    chmod +x upload_to_gdrive.sh upload_to_s3.sh create_torrent.sh

    print_success "Upload scripts created:"
    echo "  📤 upload_to_gdrive.sh - Upload to Google Drive"
    echo "  📤 upload_to_s3.sh - Upload to AWS S3"
    echo "  📤 create_torrent.sh - Create torrent for P2P sharing"
    echo ""
}

# Function to provide sharing recommendations
provide_recommendations() {
    print_header "Sharing Recommendations"

    echo "💡 Best sharing methods based on package size:"
    echo ""

    # Check package size if it exists
    if [ -n "$(ls shared_assets_* 2>/dev/null)" ]; then
        for package in shared_assets_*; do
            if [ -d "$package" ]; then
                size_bytes=$(du -sb "$package" | awk '{print $1}')
                size_human=$(du -sh "$package" | awk '{print $1}')

                echo "📦 Package: $package ($size_human)"

                if [ "$size_bytes" -lt 1073741824 ]; then  # < 1GB
                    echo "  ✅ Email attachment (if < 25MB)"
                    echo "  ✅ Google Drive / Dropbox"
                    echo "  ✅ GitHub LFS (if < 100MB)"
                elif [ "$size_bytes" -lt 5368709120 ]; then  # < 5GB
                    echo "  ✅ Google Drive / Dropbox"
                    echo "  ✅ AWS S3 with presigned URLs"
                    echo "  ⚠️  Consider splitting into smaller chunks"
                else  # > 5GB
                    echo "  ✅ AWS S3 or Google Cloud Storage"
                    echo "  ✅ BitTorrent (for very large files)"
                    echo "  ✅ Direct server transfer (rsync/scp)"
                    echo "  ⚠️  Definitely split into smaller chunks"
                fi
                echo ""
            fi
        done
    fi

    echo "🔒 Security considerations:"
    echo "  • Remove sensitive data (API keys, passwords) before sharing"
    echo "  • Consider encrypting large packages"
    echo "  • Use secure sharing methods for production data"
    echo ""

    echo "📋 What to share separately:"
    echo "  • Environment configuration (.env template)"
    echo "  • Setup documentation"
    echo "  • API keys and credentials (securely)"
    echo "  • Infrastructure setup scripts"
    echo ""
}

# Main execution
main() {
    echo "🎯 What would you like to do?"
    echo ""
    echo "1. Check available assets"
    echo "2. Create sharing package"
    echo "3. Create upload scripts"
    echo "4. Show sharing recommendations"
    echo "5. Do everything"
    echo ""

    read -p "Enter your choice (1-5): " choice

    case $choice in
        1)
            check_assets
            ;;
        2)
            check_assets
            create_sharing_package
            ;;
        3)
            create_upload_script
            ;;
        4)
            provide_recommendations
            ;;
        5)
            check_assets
            create_sharing_package
            create_upload_script
            provide_recommendations
            ;;
        *)
            print_error "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
}

# Run main function
main
