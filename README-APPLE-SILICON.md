# Using CX Consulting AI on Apple Silicon

This guide provides instructions for setting up and running the CX Consulting AI on Apple Silicon Macs (M1/M2/M3 series). The system is optimized to leverage the powerful MPS (Metal Performance Shaders) for accelerated inference on Apple Silicon.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [LLM Options for Apple Silicon](#llm-options-for-apple-silicon)
4. [Setup with Ollama (Recommended)](#setup-with-ollama-recommended)
5. [Setup with vLLM](#setup-with-vllm)
6. [Optimizing Performance](#optimizing-performance)
7. [Troubleshooting](#troubleshooting)

## Requirements

- macOS 12.0+ (Monterey or later)
- Apple Silicon Mac (M1, M2, M3 series)
- Python 3.9+
- 16GB+ RAM recommended (8GB may work with smaller models)
- 10GB+ free disk space

## Installation

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

4. Test MPS availability:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## LLM Options for Apple Silicon

For Apple Silicon Macs, you have two main options for running LLMs:

### 1. Ollama (Recommended for ease of use)

[Ollama](https://ollama.ai/) provides optimized models for macOS with an easy-to-use interface. 

**Pros:**
- Easy to install and use
- Optimized for macOS
- Many models available
- No Python dependencies needed

**Cons:**
- Less customization options
- Slightly lower performance than vLLM in some cases

### 2. vLLM (Recommended for performance)

[vLLM](https://github.com/vllm-project/vllm) is a high-performance LLM inference engine.

**Pros:**
- Higher throughput
- Better memory utilization
- More configuration options

**Cons:**
- More complex setup
- Requires PyTorch with MPS support

## Setup with Ollama (Recommended)

Ollama provides the easiest way to run LLMs on your Mac:

1. Install Ollama from [ollama.ai](https://ollama.ai/)

2. Use our setup script to download and configure a model:
```bash
# List available models
python app/scripts/setup_llm.py --list

# Download and set up the Mistral 7B model
python app/scripts/setup_llm.py --model mistral --backend ollama

# Download and set up the Gemma 7B model
python app/scripts/setup_llm.py --model gemma --backend ollama

# Download and set up the Qwen 7B model
python app/scripts/setup_llm.py --model qwen --backend ollama
```

3. Update your `.env` file to use Ollama backend:
```
LLM_BACKEND=ollama
MODEL_ID=mistral  # or gemma, qwen, etc.
OLLAMA_BASE_URL=http://localhost:11434
```

## Setup with vLLM

For higher performance, you can use vLLM:

1. Install vLLM with MPS support:
```bash
pip install vllm
```

2. Use our setup script to configure a model:
```bash
# Check your environment
python app/scripts/setup_llm.py --check

# Set up Mistral 7B with vLLM
python app/scripts/setup_llm.py --model mistral --backend vllm

# Set up Gemma 7B with vLLM
python app/scripts/setup_llm.py --model gemma --backend vllm
```

3. Update your `.env` file:
```
LLM_BACKEND=vllm
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2  # or google/gemma-7b-it
```

## Optimizing Performance

### Memory Management

For optimal performance on Apple Silicon:

1. Close unnecessary applications
2. Set appropriate chunk sizes in `.env`:
```
MAX_CHUNK_SIZE=384
CHUNK_OVERLAP=32
MAX_CHUNK_LENGTH_TOKENS=256
MAX_DOCUMENTS_PER_QUERY=3
```

3. If using vLLM, adjust quantization:
```
# Add to your .env file:
VLLM_QUANTIZATION=4bit  # Options: 4bit, 8bit
```

### Apple-Optimized Models

Some models have versions specifically optimized for Apple Silicon:

1. Modular Mistral: [mlx-community/Mistral-7B-Instruct-v0.2-mlx](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-mlx)
2. Modular Gemma: [mlx-community/gemma-7b-it](https://huggingface.co/mlx-community/gemma-7b-it)

To use these, you'll need to install the Apple MLX framework:
```bash
pip install mlx
```

## Troubleshooting

### Out of Memory Errors

If you encounter memory issues:

1. Reduce `MAX_DOCUMENTS_PER_QUERY` in `.env`
2. Use a smaller model (7B instead of larger ones)
3. Enable quantization for vLLM
4. Switch to Ollama which manages memory better

### Slow Performance

To improve performance:

1. Check that MPS is enabled
2. Use more aggressive chunking settings
3. Ensure you're using a model optimized for Apple Silicon
4. Close other applications using GPU resources

### Application Won't Start

If the application fails to start:

1. Check Python paths:
```bash
which python
echo $PYTHONPATH
```

2. Verify environment:
```bash
python app/scripts/setup_llm.py --check
```

3. Try starting with debug output:
```bash
LOG_LEVEL=DEBUG python run.py
``` 