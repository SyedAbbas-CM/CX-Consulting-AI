#!/bin/bash

# Simple Vectorstore Downloader
# Downloads vectorstore from Google Drive and places it correctly

set -e

echo "📦 Downloading CX Consulting AI Vector Database..."

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "📦 Installing gdown..."
    pip install gdown
fi

# Create directory if it doesn't exist
mkdir -p app/data

# Download vectorstore from Google Drive
echo "⬇️ Downloading vectorstore (116MB)..."
# Replace with your actual Google Drive file ID
gdown "1your_vectorstore_file_id_here" -O vectorstore.zip

# Extract to correct location
echo "📁 Extracting vectorstore..."
unzip -q vectorstore.zip -d app/data/
rm vectorstore.zip

echo "✅ Vectorstore downloaded and extracted to app/data/vectorstore/"
echo "🚀 You can now start the application with your existing models!"
