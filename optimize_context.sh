#!/bin/bash

# Optimize CX Consulting AI for T4 16GB GPU
echo "⚡ Optimizing context length for T4 16GB GPU..."

# Backup current .env
cp .env .env.backup

# Update context length and GPU settings
echo "📝 Updating .env configuration..."

# Remove old context settings if they exist
sed -i '/MAX_MODEL_LEN/d' .env
sed -i '/N_CTX/d' .env
sed -i '/GPU_LAYERS/d' .env
sed -i '/N_GPU_LAYERS/d' .env

# Add optimized settings for T4 16GB
cat >> .env << EOF

# ===== T4 16GB GPU OPTIMIZATIONS =====
# Increase context length significantly
MAX_MODEL_LEN=32768
N_CTX=32768

# Use all GPU layers for 12B model on 16GB VRAM
N_GPU_LAYERS=-1

# GPU memory optimizations
GPU_MEMORY_UTILIZATION=0.9
CUDA_VISIBLE_DEVICES=0

# Batch processing optimizations
N_BATCH=512
N_THREADS=8

# Context optimization
CONTEXT_WINDOW=32768
MAX_TOKENS_DELIVERABLE=8000
EOF

echo "✅ Context length optimized!"
echo ""
echo "📊 New Settings:"
echo "  Context Length: 32,768 tokens (4x increase)"
echo "  GPU Layers: All (-1 = full GPU acceleration)"
echo "  Max Deliverable: 8,000 tokens"
echo "  Batch Size: 512"
echo ""
echo "🔄 Restart your service to apply changes:"
echo "  sudo systemctl restart cx-consulting-ai"
