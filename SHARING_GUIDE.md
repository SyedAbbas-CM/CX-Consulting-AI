# CX Consulting AI - Complete Sharing Guide

## 📦 What You're Getting

This package contains everything needed to run the CX Consulting AI application:

- **6 LLM Models** (32GB total):
  - Gemma 2B (1.5GB) - Fast, good for testing
  - Gemma 4B (2.3GB) - Balanced performance
  - Gemma 7B (5.0GB) - Good quality/speed balance
  - Gemma 12B (6.4GB) - High quality responses
  - Gemma 27B (14GB) - Maximum quality
  - Qwen 4B (2.3GB) - Alternative model
- **Vector Store** (~300MB) - Pre-embedded documents
- **Templates & Data** - Ready-to-use configurations
- **User Database** - Test users already set up

## 🚀 Quick Start Guide

### Step 1: Get the Repository
```bash
git clone <your-repository-url>
cd CX-Consulting-AI
```

### Step 2: Extract Shared Assets
Extract the `shared_assets_*` package you received into the project directory.

### Step 3: Run the Setup Script
```bash
# Make setup script executable
chmod +x setup_shared_assets.sh

# Run the automated setup
./setup_shared_assets.sh
```

This script will:
- ✅ Check system requirements
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Copy models and vector store to correct locations
- ✅ Create and configure `.env` file
- ✅ Run basic tests

### Step 4: Start the Application
```bash
# Activate the virtual environment
source venv/bin/activate

# Start the backend
python start.py
```

The application will be available at: http://localhost:8000

## 💻 System Requirements

### Minimum Requirements:
- **OS**: macOS, Linux, or Windows with WSL
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 35GB free space
- **CPU**: Multi-core processor recommended

### Recommended for Best Performance:
- **RAM**: 32GB+ (for large models)
- **Storage**: SSD with 50GB+ free space
- **CPU**: Modern multi-core processor (8+ cores)

## 🎯 Choosing the Right Model

The setup script will automatically configure the smallest model (Gemma 2B) by default. You can change this:

### Edit `.env` file:
```bash
# For fast testing (4-6GB RAM needed)
MODEL_PATH=models/gemma-2b-it.Q4_K_M.gguf

# For balanced performance (8-12GB RAM needed)
MODEL_PATH=models/gemma-7b-it.Q4_K_M.gguf

# For high quality (12-16GB RAM needed)
MODEL_PATH=models/gemma-12B-it.QAT-Q4_0.gguf

# For maximum quality (20-32GB RAM needed)
MODEL_PATH=models/gemma-3-27B-it-QAT-Q4_0.gguf
```

## 📁 How to Share the Package

### Option 1: Cloud Storage (Recommended)
```bash
# Upload to Google Drive
./upload_to_gdrive.sh shared_assets_*

# Or upload to AWS S3
./upload_to_s3.sh shared_assets_*
```

### Option 2: Direct Transfer
```bash
# Create compressed archive
tar -czf cx_consulting_ai_assets.tar.gz shared_assets_*

# Transfer via SCP/rsync to another server
scp cx_consulting_ai_assets.tar.gz user@remote-server:/path/
```

### Option 3: BitTorrent (For Very Large Files)
```bash
# Create torrent file
./create_torrent.sh shared_assets_*

# Share the .torrent file
```

## 🛠️ Manual Setup (If Script Fails)

If the automated setup doesn't work, here's the manual process:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Directory Structure
```bash
mkdir -p models
mkdir -p app/data/vectorstore
mkdir -p app/data/templates
mkdir -p app/data/projects
mkdir -p app/data/documents
```

### 4. Copy Shared Assets
```bash
# Copy models
cp -r shared_assets_*/models/* ./models/

# Copy vector store and data
cp -r shared_assets_*/app/data/* ./app/data/
```

### 5. Create `.env` File
```bash
cp .env.example .env
# Edit .env with your model preferences
```

## 🔧 Configuration Options

### Environment Variables (`.env`):
```bash
# Model Configuration
MODEL_PATH=models/gemma-2b-it.Q4_K_M.gguf
MODEL_TYPE=llama
MAX_TOKENS=2048
TEMPERATURE=0.7

# Database
DATABASE_URL=sqlite:///./app/data/users.db

# Vector Store
VECTOR_STORE_PATH=app/data/vectorstore

# API Configuration
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

## 👥 Test Users Available

The package includes pre-configured test users:
- **testadmin** / password123 (Admin)
- **testuser** / password123 (Regular user)
- **testuser2@test.com** / password123 (Email login)

## 🚨 Troubleshooting

### Common Issues:

**1. "Model not found" error:**
```bash
# Check if model file exists
ls -la models/
# Update MODEL_PATH in .env to correct file
```

**2. "Out of memory" error:**
```bash
# Switch to smaller model in .env
MODEL_PATH=models/gemma-2b-it.Q4_K_M.gguf
```

**3. "Vector store not found:**
```bash
# Check vector store directory
ls -la app/data/vectorstore/
# Ensure files were copied correctly
```

**4. Dependencies issues:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

**5. Permission errors:**
```bash
# Fix file permissions
chmod -R 755 models/
chmod -R 755 app/data/
```

### Performance Optimization:

**For slower systems:**
- Use Gemma 2B model (fastest)
- Close other applications
- Ensure SSD storage if possible

**For faster systems:**
- Use Gemma 12B or 27B for better quality
- Increase MAX_TOKENS for longer responses
- Consider running on GPU if supported

## 📊 Expected Performance

| Model | Size | RAM Needed | Speed | Quality |
|-------|------|------------|-------|---------|
| Gemma 2B | 1.5GB | 4-6GB | Fast | Good |
| Gemma 4B | 2.3GB | 6-8GB | Medium | Better |
| Gemma 7B | 5.0GB | 8-12GB | Medium | High |
| Gemma 12B | 6.4GB | 12-16GB | Slow | Very High |
| Gemma 27B | 14GB | 20-32GB | Very Slow | Excellent |

## 🔄 Updating Models

To add new models later:
1. Download GGUF format models
2. Place in `models/` directory
3. Update `MODEL_PATH` in `.env`
4. Restart the application

## 📞 Getting Help

1. **Check logs**: `tail -f app.log`
2. **Verify setup**: Run `./setup_shared_assets.sh` again
3. **Test basic functionality**: Try different models
4. **Check system resources**: Monitor RAM and CPU usage

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ Application starts without errors
- ✅ Web interface loads at http://localhost:8000
- ✅ You can log in with test users
- ✅ AI responses are generated successfully
- ✅ Vector search returns relevant results

---

**Need more help?** Check the main project README or contact the project maintainer.
