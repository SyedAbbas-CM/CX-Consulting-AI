# CX Consulting AI - Setup for Tariq

## 🎯 Super Simple Setup (3 steps)

### Step 1: Get the Code
```bash
git clone https://github.com/Cloud-Primero/CX-Consulting-AI.git
cd CX-Consulting-AI
```

### Step 2: Download Models & Setup Everything
```bash
./get_models.sh
```
That's it! This script:
- Downloads all models from Google Drive
- Sets up the vector store
- Creates virtual environment
- Installs all dependencies
- Configures everything

### Step 3: Start the App
```bash
source venv/bin/activate
python start.py
```

## 🌐 Access the App
- **URL**: http://localhost:8000
- **Username**: azureuser
- **Password**: demo123456

## 📦 What You Get
- **6 AI Models** (2B to 27B parameters)
- **Vector Store** (pre-loaded documents)
- **Ready-to-use setup**

## 💻 Requirements
- **Python 3.8+**
- **8GB RAM minimum**
- **30GB free space**
- **Internet connection** (for download)

## 🚨 If Something Goes Wrong
1. Make sure you have Python 3.8+: `python3 --version`
2. Make sure you have internet connection
3. Check if you have enough disk space: `df -h`

That's it! No complicated setup, no manual file copying, just run one script and you're done.

---
**Contact**: Message if you need the Google Drive links updated in the script
