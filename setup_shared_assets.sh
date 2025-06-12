#!/bin/bash

# CX Consulting AI - Shared Assets Setup Script
# For recipients of shared LLM models and vector store

set -e

echo "🚀 CX Consulting AI - Shared Assets Setup"
echo "========================================"
echo ""

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

# Function to check system requirements
check_requirements() {
    print_header "Checking System Requirements"

    # Check Python
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version | awk '{print $2}')
        print_success "Python 3 found: $python_version"
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi

    # Check Git
    if command -v git &> /dev/null; then
        print_success "Git found"
    else
        print_warning "Git not found. You may need it to clone the repository."
    fi

    # Check available disk space
    available_space=$(df -h . | awk 'NR==2 {print $4}')
    print_success "Available disk space: $available_space"

    # Check RAM
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        total_ram=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2, $3}')
        print_success "Total RAM: $total_ram"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        total_ram=$(free -h | awk '/^Mem:/ {print $2}')
        print_success "Total RAM: $total_ram"
    fi

    echo ""
}

# Function to find and validate shared assets
find_shared_assets() {
    print_header "Looking for Shared Assets"

    # Look for shared asset directories
    shared_dirs=($(ls -d shared_assets_* 2>/dev/null || echo ""))

    if [ ${#shared_dirs[@]} -eq 0 ]; then
        print_error "No shared asset directories found!"
        echo "Please ensure you have extracted/downloaded the shared assets in this directory."
        echo "Looking for directories named: shared_assets_*"
        exit 1
    fi

    # Select the most recent one if multiple exist
    latest_dir=$(ls -dt shared_assets_* | head -1)
    print_success "Found shared assets: $latest_dir"

    # Validate contents
    echo "📋 Validating package contents:"

    if [ -d "$latest_dir/models" ]; then
        model_count=$(find "$latest_dir/models" -name "*.gguf" | wc -l)
        print_success "Models directory found ($model_count GGUF files)"
    else
        print_warning "No models directory found"
    fi

    if [ -d "$latest_dir/app/data/vectorstore" ]; then
        vectorstore_size=$(du -sh "$latest_dir/app/data/vectorstore" | awk '{print $1}')
        print_success "Vector store found (Size: $vectorstore_size)"
    else
        print_warning "No vector store found"
    fi

    if [ -f "$latest_dir/SETUP_INSTRUCTIONS.md" ]; then
        print_success "Setup instructions found"
    fi

    echo ""
    echo "📦 Using package: $latest_dir"
    echo ""

    export SHARED_ASSETS_DIR="$latest_dir"
}

# Function to setup the main repository
setup_repository() {
    print_header "Setting Up Main Repository"

    if [ ! -f "requirements.txt" ]; then
        print_warning "No requirements.txt found. This doesn't look like the main CX Consulting AI repository."
        echo ""
        echo "Please ensure you are in the main project directory, or clone it first:"
        echo "git clone <repository-url>"
        echo ""
        read -p "Continue anyway? (y/N): " continue_setup
        if [[ ! "$continue_setup" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Create necessary directories for LLMs and vector store only
    print_success "Creating directory structure..."
    mkdir -p models
    mkdir -p app/data/vectorstore

    # Copy shared assets - ONLY models and vector store
    if [ -n "$SHARED_ASSETS_DIR" ]; then
        print_success "Copying shared assets..."

        # Copy models
        if [ -d "$SHARED_ASSETS_DIR/models" ]; then
            cp -r "$SHARED_ASSETS_DIR/models"/* ./models/ 2>/dev/null || true
            print_success "✅ LLM models copied"
        else
            print_warning "No models found in shared assets"
        fi

        # Copy vector store only
        if [ -d "$SHARED_ASSETS_DIR/app/data/vectorstore" ]; then
            cp -r "$SHARED_ASSETS_DIR/app/data/vectorstore"/* ./app/data/vectorstore/ 2>/dev/null || true
            print_success "✅ Vector store copied"
        else
            print_warning "No vector store found in shared assets"
        fi

        print_success "🎯 LLMs and vector store setup complete!"
        echo "ℹ️  Note: Only models and vector store were copied."
        echo "   You'll need to set up users and other data separately."
    fi

    echo ""
}

# Function to setup Python environment
setup_python_env() {
    print_header "Setting Up Python Environment"

    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_success "Creating virtual environment..."
        python3 -m venv venv
    else
        print_success "Virtual environment already exists"
    fi

    # Activate virtual environment
    print_success "Activating virtual environment..."
    source venv/bin/activate

    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_success "Installing requirements..."
        pip install -r requirements.txt
    else
        print_warning "No requirements.txt found. Installing basic dependencies..."
        pip install fastapi uvicorn sqlalchemy chromadb llama-cpp-python python-multipart python-jose[cryptography] passlib[bcrypt] redis
    fi

    echo ""
}

# Function to setup configuration
setup_configuration() {
    print_header "Setting Up Configuration"

    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_success "Creating .env from .env.example..."
            cp .env.example .env
        else
            print_success "Creating basic .env file..."
            cat > .env << 'EOF'
# CX Consulting AI Configuration

# Model Configuration
MODEL_PATH=models/your-model.gguf
MODEL_TYPE=llama
MAX_TOKENS=2048
TEMPERATURE=0.7

# Database
DATABASE_URL=sqlite:///./app/data/users.db

# Vector Store
VECTOR_STORE_PATH=app/data/vectorstore

# API Configuration
SECRET_KEY=your-secret-key-change-this
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000
EOF
        fi
    else
        print_success ".env file already exists"
    fi

    # Update model path in .env if we found models
    if [ -d "models" ] && [ "$(ls -A models)" ]; then
        first_model=$(ls models/*.gguf 2>/dev/null | head -1)
        if [ -n "$first_model" ]; then
            # Update MODEL_PATH in .env
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s|MODEL_PATH=.*|MODEL_PATH=$first_model|" .env
            else
                sed -i "s|MODEL_PATH=.*|MODEL_PATH=$first_model|" .env
            fi
            print_success "Updated MODEL_PATH in .env to: $first_model"
        fi
    fi

    echo ""
    print_warning "Please review and update .env file with your specific settings:"
    echo "  • Model paths"
    echo "  • API keys (if needed)"
    echo "  • Database settings"
    echo "  • Secret keys"
    echo ""
}

# Function to run basic tests
run_tests() {
    print_header "Running Basic Tests"

    # Test model loading
    if [ -f "models"/*.gguf ]; then
        print_success "Model files found - ready for loading"
    else
        print_warning "No model files found in models directory"
    fi

    # Test vector store
    if [ -d "app/data/vectorstore" ] && [ "$(ls -A app/data/vectorstore)" ]; then
        print_success "Vector store data found"
    else
        print_warning "No vector store data found"
    fi

    # Test Python imports (basic)
    if source venv/bin/activate && python3 -c "import fastapi, uvicorn, sqlalchemy" 2>/dev/null; then
        print_success "Basic Python dependencies working"
    else
        print_warning "Some Python dependencies may have issues"
    fi

    echo ""
}

# Function to show next steps
show_next_steps() {
    print_header "Next Steps"

    echo "🎉 Setup complete! Here's what to do next:"
    echo ""
    echo "1. 📝 Review the configuration:"
    echo "   nano .env"
    echo ""
    echo "2. 🚀 Start the application:"
    echo "   source venv/bin/activate"
    echo "   python start.py"
    echo ""
    echo "3. 🌐 Open your browser to:"
    echo "   http://localhost:8000"
    echo ""
    echo "4. 📚 If you need help:"
    echo "   • Check app.log for error messages"
    echo "   • Review the SETUP_INSTRUCTIONS.md in the shared assets"
    echo "   • Ensure all file paths in .env are correct"
    echo ""

    # Show resource usage estimate
    if [ -d "models" ]; then
        model_size=$(du -sh models 2>/dev/null | awk '{print $1}' || echo "Unknown")
        echo "💾 Estimated RAM usage: ~${model_size} (for model) + 1-2GB (for application)"
    fi

    echo ""
    print_success "You're all set! 🚀"
}

# Main execution
main() {
    check_requirements
    find_shared_assets
    setup_repository
    setup_python_env
    setup_configuration
    run_tests
    show_next_steps
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
