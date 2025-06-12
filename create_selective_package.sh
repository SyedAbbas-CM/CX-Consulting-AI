#!/bin/bash

# CX Consulting AI - Selective Package Creator
# Create smaller packages by selecting specific models

set -e

echo "🎯 CX Consulting AI - Selective Package Creator"
echo "=============================================="
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

# Function to list available models
list_models() {
    print_header "Available Models"

    if [ ! -d "models/" ]; then
        print_error "No models directory found!"
        exit 1
    fi

    echo "📋 Models available for packaging:"
    echo ""

    declare -a model_files
    declare -a model_sizes
    model_index=1

    for model in models/*.gguf; do
        if [ -f "$model" ]; then
            size=$(ls -lh "$model" | awk '{print $5}')
            model_name=$(basename "$model")

            echo "  $model_index. $model_name ($size)"
            model_files[$model_index]="$model"
            model_sizes[$model_index]="$size"
            ((model_index++))
        fi
    done

    if [ ${#model_files[@]} -eq 0 ]; then
        print_error "No GGUF model files found in models directory!"
        exit 1
    fi

    echo ""
    export model_files
    export model_sizes
    return $((model_index - 1))
}

# Function to create package presets
show_presets() {
    print_header "Package Presets"

    echo "💡 Recommended package combinations:"
    echo ""
    echo "A. 🔥 Small & Fast (Recommended for testing)"
    echo "   • Gemma 2B (1.5GB) - Fast inference, good for demos"
    echo "   • Vector store + templates"
    echo "   • Total: ~2GB"
    echo ""
    echo "B. 🎯 Balanced Performance"
    echo "   • Gemma 7B (5.0GB) - Good balance of speed and quality"
    echo "   • Vector store + templates"
    echo "   • Total: ~5.5GB"
    echo ""
    echo "C. 🚀 High Performance"
    echo "   • Gemma 12B (6.4GB) - Better quality, slower inference"
    echo "   • Vector store + templates"
    echo "   • Total: ~7GB"
    echo ""
    echo "D. 🏆 Maximum Quality"
    echo "   • Gemma 27B (14GB) - Best quality, requires more RAM"
    echo "   • Vector store + templates"
    echo "   • Total: ~14.5GB"
    echo ""
    echo "E. 🎲 Custom Selection"
    echo "   • Choose specific models yourself"
    echo ""
}

# Function to select models
select_models() {
    local total_models=$1
    declare -a selected_models

    show_presets

    echo "🎯 What would you like to package?"
    echo ""
    echo "A. Small & Fast (Gemma 2B)"
    echo "B. Balanced Performance (Gemma 7B)"
    echo "C. High Performance (Gemma 12B)"
    echo "D. Maximum Quality (Gemma 27B)"
    echo "E. Custom Selection"
    echo "F. All models (32GB package)"
    echo ""

    read -p "Enter your choice (A-F): " preset_choice

    case $preset_choice in
        A|a)
            selected_models=("models/gemma-2b-it.Q4_K_M.gguf")
            package_suffix="small_fast"
            ;;
        B|b)
            selected_models=("models/gemma-7b-it.Q4_K_M.gguf")
            package_suffix="balanced"
            ;;
        C|c)
            selected_models=("models/gemma-12B-it.QAT-Q4_0.gguf")
            package_suffix="high_performance"
            ;;
        D|d)
            selected_models=("models/gemma-3-27B-it-QAT-Q4_0.gguf")
            package_suffix="maximum_quality"
            ;;
        E|e)
            echo ""
            echo "📋 Select models to include (enter numbers separated by spaces):"
            echo "   Example: 1 3 5"
            echo ""
            read -p "Model numbers: " model_numbers

            for num in $model_numbers; do
                if [ "$num" -ge 1 ] && [ "$num" -le "$total_models" ]; then
                    selected_models+=("${model_files[$num]}")
                else
                    print_warning "Invalid model number: $num (ignored)"
                fi
            done
            package_suffix="custom"
            ;;
        F|f)
            for i in $(seq 1 $total_models); do
                selected_models+=("${model_files[$i]}")
            done
            package_suffix="all_models"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac

    if [ ${#selected_models[@]} -eq 0 ]; then
        print_error "No models selected!"
        exit 1
    fi

    export selected_models
    export package_suffix
}

# Function to create the selective package
create_package() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local package_dir="shared_assets_${package_suffix}_${timestamp}"

    print_header "Creating Selective Package"

    mkdir -p "$package_dir"
    echo "📦 Creating package: $package_dir"
    echo ""

    # Create models directory and copy selected models
    mkdir -p "$package_dir/models"

    echo "📄 Including models:"
    total_model_size=0

    for model in "${selected_models[@]}"; do
        if [ -f "$model" ]; then
            model_name=$(basename "$model")
            model_size_mb=$(du -m "$model" | awk '{print $1}')
            model_size_human=$(ls -lh "$model" | awk '{print $5}')

            echo "  ✅ $model_name ($model_size_human)"
            cp "$model" "$package_dir/models/"
            total_model_size=$((total_model_size + model_size_mb))
        else
            print_warning "Model not found: $model"
        fi
    done

    # Copy vector store if it exists
    if [ -d "app/data/vectorstore/" ] && [ "$(ls -A app/data/vectorstore/)" ]; then
        print_success "Including vector store..."
        mkdir -p "$package_dir/app/data/"
        cp -r app/data/vectorstore/ "$package_dir/app/data/"
    fi

    # Copy other important data
    if [ -d "app/data/templates/" ]; then
        print_success "Including templates..."
        mkdir -p "$package_dir/app/data/"
        cp -r app/data/templates/ "$package_dir/app/data/"
    fi

    if [ -f "app/data/users.db" ]; then
        print_success "Including user database..."
        mkdir -p "$package_dir/app/data/"
        cp app/data/users.db "$package_dir/app/data/"
    fi

    # Create setup instructions with model-specific information
    cat > "$package_dir/SETUP_INSTRUCTIONS.md" << EOF
# CX Consulting AI - Selective Package Setup

## Package Information
- Package Type: $package_suffix
- Created: $(date)
- Number of Models: ${#selected_models[@]}

## Included Models:
EOF

    for model in "${selected_models[@]}"; do
        if [ -f "$model" ]; then
            model_name=$(basename "$model")
            model_size=$(ls -lh "$model" | awk '{print $5}')

            # Estimate RAM requirements based on model size
            case $model_name in
                *2b*|*2B*)
                    ram_estimate="4-6GB RAM recommended"
                    performance="Fast inference, good for demos"
                    ;;
                *4b*|*4B*)
                    ram_estimate="6-8GB RAM recommended"
                    performance="Good balance of speed and quality"
                    ;;
                *7b*|*7B*)
                    ram_estimate="8-12GB RAM recommended"
                    performance="Balanced performance"
                    ;;
                *12B*|*12b*)
                    ram_estimate="12-16GB RAM recommended"
                    performance="High quality responses"
                    ;;
                *27B*|*27b*)
                    ram_estimate="20-32GB RAM recommended"
                    performance="Maximum quality, requires powerful hardware"
                    ;;
                *)
                    ram_estimate="8GB+ RAM recommended"
                    performance="Performance varies by model"
                    ;;
            esac

            cat >> "$package_dir/SETUP_INSTRUCTIONS.md" << EOF
- **$model_name** ($model_size)
  - RAM: $ram_estimate
  - Performance: $performance

EOF
        fi
    done

    cat >> "$package_dir/SETUP_INSTRUCTIONS.md" << 'EOF'

## Quick Setup Steps:

### 1. Clone the main repository:
```bash
git clone <your-repo-url>
cd CX-Consulting-AI
```

### 2. Run the setup script:
```bash
# Extract your shared package to the project directory
# Then run:
./setup_shared_assets.sh
```

### 3. Start the application:
```bash
source venv/bin/activate
python start.py
```

## Model Configuration:
The setup script will automatically configure the first model as default.
To use a different model, edit the `.env` file:

```bash
# Edit .env file
MODEL_PATH=models/your-preferred-model.gguf
```

## Performance Notes:
- Smaller models (2B-4B): Fast responses, good for development
- Medium models (7B-12B): Balanced quality and speed
- Large models (27B+): Best quality but requires more resources

## Troubleshooting:
1. Ensure your system has enough RAM for the selected model
2. Check that all file paths in .env are correct
3. Review app.log for any error messages
4. Make sure all dependencies are installed correctly

For more help, check the main project documentation.
EOF

    # Create a manifest file
    cat > "$package_dir/MANIFEST.txt" << EOF
CX Consulting AI - Selective Package
Package Type: $package_suffix
Created: $(date)

Models Included (${#selected_models[@]}):
EOF

    for model in "${selected_models[@]}"; do
        if [ -f "$model" ]; then
            model_name=$(basename "$model")
            model_size=$(ls -lh "$model" | awk '{print $5}')
            echo "  - $model_name ($model_size)" >> "$package_dir/MANIFEST.txt"
        fi
    done

    cat >> "$package_dir/MANIFEST.txt" << EOF

Additional Components:
  - Vector store (ChromaDB with embedded documents)
  - Templates and configuration files
  - User database (development users)
  - Setup instructions and documentation

Total Package Size: $(du -sh "$package_dir" | awk '{print $1}')

Installation:
1. Extract to your CX Consulting AI project directory
2. Run ./setup_shared_assets.sh
3. Start the application with: python start.py
EOF

    # Calculate and display package size
    package_size=$(du -sh "$package_dir" | awk '{print $1}')

    print_success "Package created successfully!"
    echo ""
    echo "📦 Package Details:"
    echo "   Name: $package_dir"
    echo "   Size: $package_size"
    echo "   Models: ${#selected_models[@]}"
    echo "   Type: $package_suffix"
    echo ""

    print_header "Recommended Sharing Method"

    # Convert size to MB for comparison
    package_size_mb=$(du -m "$package_dir" | awk '{print $1}')

    if [ "$package_size_mb" -lt 1000 ]; then  # < 1GB
        echo "📤 Recommended: Google Drive, Dropbox, or WeTransfer"
        echo "🔗 This package is small enough for most cloud services"
    elif [ "$package_size_mb" -lt 5000 ]; then  # < 5GB
        echo "📤 Recommended: Google Drive, Dropbox (premium), or AWS S3"
        echo "⚠️  May need to compress or split for some services"
    else  # > 5GB
        echo "📤 Recommended: AWS S3, Google Cloud Storage, or BitTorrent"
        echo "⚠️  Consider creating multiple smaller packages"
    fi

    echo ""
    echo "💡 Next steps:"
    echo "   1. Test the package locally first"
    echo "   2. Use upload scripts: ./upload_to_gdrive.sh $package_dir"
    echo "   3. Share setup_shared_assets.sh with your recipient"
    echo "   4. Provide them with the repository URL"
    echo ""
}

# Main execution
main() {
    list_models
    total_models=$?
    select_models $total_models
    create_package
}

main "$@"
