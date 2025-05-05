# Models Directory

This directory stores local models used by the CX Consulting Agent.

## Embedding Models

The application uses embedding models for the RAG (Retrieval-Augmented Generation) system. By default, the system is configured to use the BAAI/bge-small-en-v1.5 model, which is optimized for RAG applications.

### Using ONNX Acceleration

For improved performance, you can convert the embedding model to ONNX format using the provided script:

```bash
# Convert default BGE model to ONNX
python -m app.scripts.convert_to_onnx

# Convert with quantization for smaller size
python -m app.scripts.convert_to_onnx --quantize

# Convert a different model
python -m app.scripts.convert_to_onnx --model BAAI/bge-base-en-v1.5 --output models/bge-base-en-v1.5.onnx
```

After conversion, update your `.env` file to use the ONNX model:

```
EMBEDDING_TYPE=onnx
ONNX_MODEL_PATH=models/bge-small-en-v1.5.onnx
ONNX_TOKENIZER=BAAI/bge-small-en-v1.5
```

## LLM Models

For local LLM inference, place your model files in this directory. Supported formats vary based on the backend:

### llama.cpp Backend

```
# GGUF format models
models/gemma-2b-it.Q4_K_M.gguf
models/mistral-7b-instruct-v0.2.Q5_K_M.gguf
```

### vLLM Backend

```
# Hugging Face models
models/NousResearch/Nous-Hermes-2-Mistral-7B-DPO
models/google/gemma-7b-it
```

### Ollama Backend

No local files needed - models are managed by Ollama directly. 