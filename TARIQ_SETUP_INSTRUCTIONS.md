# CX Consulting AI - Setup Instructions for Tariq

## 🎯 Quick Setup Guide

### Step 1: Get the Project
```bash
git clone https://github.com/Cloud-Primero/CX-Consulting-AI.git
cd CX-Consulting-AI
```

### Step 2: Get the Shared Assets
**Contact Arham for the shared assets package** (32GB with all models and data)

Options:
- Google Drive link
- Direct file transfer
- Cloud storage download

### Step 3: Run the Automated Setup
```bash
# Extract the shared assets package to the project directory
# You should have a folder like: shared_assets_YYYYMMDD_HHMMSS

# Run the setup script
chmod +x setup_shared_assets.sh
./setup_shared_assets.sh
```

### Step 4: Start the Application
```bash
# Activate the environment
source venv/bin/activate

# Start the backend
python start.py
```

### Step 5: Access the Application
Open your browser to: http://localhost:8000

## 🔐 Login Credentials
- **Username**: azureuser
- **Password**: demo123456

## 📦 What's Included
- **6 LLM Models** (2B to 27B parameters)
- **Vector Store** (pre-embedded documents)
- **Test Data** (sample projects and conversations)
- **Full Documentation**

## 💻 System Requirements
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 40GB free space
- **Python**: 3.8+
- **OS**: macOS, Linux, or Windows with WSL

## 🚨 Troubleshooting
1. **Out of memory**: Use smaller model (Gemma 2B)
2. **Dependencies issues**: Run `pip install -r requirements.txt`
3. **Port conflict**: Change PORT in .env file

## 🆘 Need Help?
- Check app.log for errors
- Review SHARING_GUIDE.md for detailed instructions
- Contact the development team

---
**Created**: June 2025
**Version**: Production v1.0
