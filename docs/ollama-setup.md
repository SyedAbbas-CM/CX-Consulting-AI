# Setting Up Ollama for CX Consulting AI on macOS

This guide provides step-by-step instructions for setting up Ollama on macOS to run LLMs locally for the CX Consulting AI system.

## What is Ollama?

[Ollama](https://ollama.ai/) is an open-source tool that simplifies running large language models locally on your computer. It provides optimized versions of popular models like Mistral, Gemma, Qwen, and Llama, with easy installation and management.

## Benefits of Using Ollama

- **Simple Setup**: One-click installation for macOS
- **Optimized Performance**: Models are optimized for Apple Silicon
- **Easy Model Management**: Simple commands to pull and run models
- **REST API**: Integration with applications through a REST API
- **No Account Needed**: Unlike Hugging Face, no login or API keys required

## System Requirements

- macOS 12.0+ (Monterey or later)
- Apple Silicon Mac (M1, M2, M3 series) or Intel Mac
- 8GB+ RAM (16GB+ recommended for larger models)
- 5GB+ free disk space per model

## Installation Steps

1. **Download Ollama**

   Download the Ollama installer from the official website:
   [https://ollama.ai/download](https://ollama.ai/download)

2. **Install Ollama**

   Open the downloaded `.dmg` file and drag the Ollama app to your Applications folder.

3. **Launch Ollama**

   Open Ollama from your Applications folder. You should see the Ollama icon appear in your menu bar.

4. **Verify Installation**

   Open Terminal and run:
   ```bash
   ollama --version
   ```
   
   You should see the Ollama version displayed.

## Pulling Models for CX Consulting AI

For the CX Consulting AI system, we recommend the following models:

### Option 1: Using Our Setup Script

The easiest way to pull models is using our setup script:

```bash
# List available models
python app/scripts/setup_llm.py --list

# Pull Mistral 7B (recommended starter model)
python app/scripts/setup_llm.py --model mistral --backend ollama

# Pull Gemma 7B
python app/scripts/setup_llm.py --model gemma --backend ollama

# Pull Qwen 7B
python app/scripts/setup_llm.py --model qwen --backend ollama

# Pull Llama 3 8B
python app/scripts/setup_llm.py --model llama3 --backend ollama
```

### Option 2: Manual Model Installation

You can also pull models manually using Ollama commands:

```bash
# Pull Mistral 7B
ollama pull mistral

# Pull Gemma 7B
ollama pull gemma

# Pull Qwen 7B
ollama pull qwen

# Pull Llama 3 8B
ollama pull llama3
```

## Configuring CX Consulting AI to Use Ollama

Edit your `.env` file in the project root directory:

```
# Change these settings
LLM_BACKEND=ollama
MODEL_ID=mistral  # or gemma, qwen, etc.
OLLAMA_BASE_URL=http://localhost:11434
```

## Testing the Setup

Run our test script to verify that Ollama is working properly:

```bash
python app/scripts/test_system.py
```

## Model Recommendations for Different System Configurations

### For 8GB RAM Systems:
- **Recommended**: Mistral 7B, Qwen 7B
- **Consider**: Using 4-bit quantized versions for better performance

### For 16GB RAM Systems:
- **Recommended**: Any 7B or 8B model (Mistral, Gemma, Qwen, Llama 3)
- **Consider**: Using chat templates with 'ollama run' for the best experience

### For 32GB+ RAM Systems:
- **Recommended**: Any model, including larger 13B or even 70B variants
- **Consider**: Running multiple models simultaneously

## Troubleshooting

### Model Downloads Failing
- Check your internet connection
- Ensure you have enough disk space
- Try running with sudo: `sudo ollama pull mistral`

### Out of Memory Errors
- Close other applications
- Try a smaller model
- Use a quantized version of the model: `ollama pull mistral:4bit`

### API Connection Issues
- Ensure Ollama is running (check menu bar icon)
- Verify the API URL in .env is correct
- Check if port 11434 is blocked by a firewall

## Advanced Usage

### Creating Model Tags with Custom Parameters

You can create custom model configurations with different parameters:

```bash
ollama create mygemma -f - << EOF
FROM gemma:7b-instruct
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.95
EOF
```

Then update your `.env` file:
```
MODEL_ID=mygemma
```

### Using Multiple Models

You can switch between models by changing the `MODEL_ID` in your `.env` file or by specifying a different model when running scripts.

### Checking Available Models

To see the models you have installed:

```bash
ollama list
```

## Additional Resources

- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [Ollama Model Library](https://ollama.ai/library)
- [Modelfiles Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) 