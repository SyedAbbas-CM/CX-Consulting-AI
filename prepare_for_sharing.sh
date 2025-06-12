#!/bin/bash

# CX Consulting AI - Prepare for Sharing
# One-click script to prepare everything for sharing

set -e

echo "🎁 CX Consulting AI - Prepare for Sharing"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${YELLOW}👉 $1${NC}"
}

# Check if we have the shared assets package
if [ ! -d "shared_assets_"* ]; then
    echo "Creating complete sharing package..."
    echo "5" | ./share_project_assets.sh > /dev/null 2>&1
fi

# Get the latest package
PACKAGE_DIR=$(ls -dt shared_assets_* | head -1)
PACKAGE_SIZE=$(du -sh "$PACKAGE_DIR" | awk '{print $1}')

print_success "Package ready: $PACKAGE_DIR ($PACKAGE_SIZE)"
echo ""

# Create a transfer bundle
echo "📦 Creating transfer bundle..."
BUNDLE_NAME="cx_consulting_ai_complete_$(date +%Y%m%d)"

# Copy everything needed
mkdir -p "$BUNDLE_NAME"
cp -r "$PACKAGE_DIR" "$BUNDLE_NAME/"
cp setup_shared_assets.sh "$BUNDLE_NAME/"
cp SHARING_GUIDE.md "$BUNDLE_NAME/"

# Create a simple start script for the recipient
cat > "$BUNDLE_NAME/START_HERE.md" << 'EOF'
# 🚀 CX Consulting AI - Getting Started

## What to do:

1. **First, clone the main repository:**
   ```bash
   git clone <your-repo-url>
   cd CX-Consulting-AI
   ```

2. **Extract this bundle into the project directory**

3. **Run the setup script:**
   ```bash
   chmod +x setup_shared_assets.sh
   ./setup_shared_assets.sh
   ```

4. **Start the application:**
   ```bash
   source venv/bin/activate
   python start.py
   ```

5. **Open your browser to:** http://localhost:8000

## Test Users:
- testadmin / password123
- testuser / password123
- testuser2@test.com / password123

## Need help?
Read the complete SHARING_GUIDE.md for detailed instructions.
EOF

# Calculate total bundle size
BUNDLE_SIZE=$(du -sh "$BUNDLE_NAME" | awk '{print $1}')

print_success "Complete bundle created: $BUNDLE_NAME ($BUNDLE_SIZE)"
echo ""

# Show sharing options
echo "🚀 Ready to Share! Here are your options:"
echo ""

print_step "Option 1: Google Drive (Easiest)"
echo "   Run: ./upload_to_gdrive.sh $BUNDLE_NAME"
echo "   Then share the Google Drive folder link"
echo ""

print_step "Option 2: AWS S3 (Fast, with direct links)"
echo "   Run: ./upload_to_s3.sh $BUNDLE_NAME"
echo "   Share the generated presigned URLs"
echo ""

print_step "Option 3: Compress and transfer manually"
echo "   Run: tar -czf ${BUNDLE_NAME}.tar.gz $BUNDLE_NAME"
echo "   Share the .tar.gz file via your preferred method"
echo ""

print_step "Option 4: Direct server transfer"
echo "   Run: scp -r $BUNDLE_NAME user@server:/path/"
echo "   Or: rsync -av $BUNDLE_NAME user@server:/path/"
echo ""

# Show what to send to recipient
echo "📧 What to send to your collaborator:"
echo ""
echo "   1. The shared package (via one of the options above)"
echo "   2. Your repository URL"
echo "   3. This message:"
echo ""
echo "   \"Hi! I've shared the CX Consulting AI models and data with you."
echo "   Please:"
echo "   1. Clone the repo from: <your-repo-url>"
echo "   2. Extract the shared bundle into the project directory"
echo "   3. Run: ./setup_shared_assets.sh"
echo "   4. Start with: python start.py"
echo "   "
echo "   Everything is automated - the setup script will handle all dependencies"
echo "   and configuration. Check START_HERE.md for quick instructions.\""
echo ""

print_info "Bundle contains:"
echo "   • All 6 LLM models (32GB)"
echo "   • Vector store with embedded documents"
echo "   • Automated setup script"
echo "   • Complete documentation"
echo "   • Test users and sample data"
echo ""

print_success "Everything ready for sharing! 🎉"
echo ""
echo "💡 Tip: Test the setup locally first by running ./setup_shared_assets.sh"
echo "    in a fresh directory to make sure everything works."
