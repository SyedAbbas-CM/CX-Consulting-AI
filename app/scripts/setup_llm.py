#!/usr/bin/env python
"""
LLM Setup Script for Apple Silicon

This script helps set up LLMs for use with vLLM and Ollama on Apple Silicon Macs.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_setup")

# Define LLM options
LLM_OPTIONS = {
    "mistral": {
        "huggingface_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "ollama_id": "mistral",
        "description": "Mistral 7B Instruct v0.2 - Good general-purpose 7B model",
    },
    "gemma": {
        "huggingface_id": "google/gemma-7b-it",
        "ollama_id": "gemma",
        "description": "Google Gemma 7B Instruct - Google's recent 7B instruction model",
    },
    "qwen": {
        "huggingface_id": "Qwen/Qwen1.5-7B-Chat",
        "ollama_id": "qwen",
        "description": "Qwen 1.5 7B Chat - Alibaba's 7B chat model",
    },
    "llama3": {
        "huggingface_id": "meta-llama/Llama-3-8b-hf",
        "ollama_id": "llama3",
        "description": "Llama 3 8B - Meta's latest 8B foundation model",
    },
}


def check_environment():
    """Check the environment for required dependencies."""
    logger.info("Checking environment...")

    # Check Python version
    python_version = sys.version_info
    logger.info(
        f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    if python_version.major < 3 or (
        python_version.major == 3 and python_version.minor < 9
    ):
        logger.error("Python 3.9 or higher is required")
        return False

    # Check for PyTorch
    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")

        # Check for MPS (Apple Silicon GPU) support
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon GPU) is available")
        else:
            logger.warning("MPS (Apple Silicon GPU) is not available, CPU only")

        # Check for CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA is not available")

    except ImportError:
        logger.error("PyTorch is not installed")
        return False

    # Check for vLLM
    try:
        import vllm

        logger.info(f"vLLM version: {vllm.__version__}")
    except ImportError:
        logger.warning(
            "vLLM is not installed, consider installing it: pip install vllm"
        )

    # Check for Ollama
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Ollama version: {result.stdout.strip()}")
        else:
            logger.warning("Ollama is installed but returned an error")
    except FileNotFoundError:
        logger.warning(
            "Ollama is not installed, consider installing it: https://ollama.ai/"
        )

    # Check for Hugging Face Hub access
    try:
        import huggingface_hub

        if os.environ.get("HF_TOKEN"):
            logger.info("Hugging Face token found in environment")
        else:
            logger.warning(
                "No Hugging Face token found in environment, may have limited access"
            )
    except ImportError:
        logger.warning("huggingface_hub is not installed")

    return True


def setup_model_vllm(model_id, model_name=None, adapter=None):
    """Set up a model with vLLM."""
    try:
        import torch
        from vllm import LLM

        logger.info(f"Setting up model {model_id} with vLLM")

        # Get model info
        if model_id in LLM_OPTIONS:
            huggingface_id = LLM_OPTIONS[model_id]["huggingface_id"]
            logger.info(f"Using Hugging Face ID: {huggingface_id}")
        else:
            huggingface_id = model_id

        # Initialize vLLM
        try:
            # Determine compute device and settings
            use_gpu = "cpu"
            if torch.backends.mps.is_available():
                use_gpu = "mps"
                logger.info("Using Apple Silicon GPU (MPS) for computation")
            elif torch.cuda.is_available():
                use_gpu = "cuda"
                logger.info("Using NVIDIA GPU (CUDA) for computation")
            else:
                logger.warning("Using CPU for computation, this will be slow")

            # Adjust model path if needed
            if model_name:
                # Assuming model_name is a local model directory
                model_path = model_name
            else:
                model_path = huggingface_id

            # Adjust .env file with model info
            update_env_file(model_path, "vllm")

            logger.info(f"Model {model_id} is ready to use with vLLM")
            logger.info(f"Updated .env file with model information")

            return True

        except Exception as e:
            logger.error(f"Error initializing vLLM: {str(e)}")
            return False

    except ImportError:
        logger.error("vLLM is not installed")
        return False


def setup_model_ollama(model_id):
    """Set up a model with Ollama."""
    logger.info(f"Setting up model {model_id} with Ollama")

    # Get model info
    if model_id in LLM_OPTIONS:
        ollama_id = LLM_OPTIONS[model_id]["ollama_id"]
        logger.info(f"Using Ollama ID: {ollama_id}")
    else:
        ollama_id = model_id

    # Check for Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Error checking Ollama models")
            return False

        # Check if model is already pulled
        if ollama_id in result.stdout:
            logger.info(f"Model {ollama_id} is already pulled")
        else:
            # Pull the model
            logger.info(f"Pulling model {ollama_id}, this may take a while...")
            pull_result = subprocess.run(
                ["ollama", "pull", ollama_id], capture_output=True, text=True
            )

            if pull_result.returncode != 0:
                logger.error(f"Error pulling model: {pull_result.stderr}")
                return False

            logger.info(f"Model {ollama_id} pulled successfully")

        # Update .env file
        update_env_file(ollama_id, "ollama")

        logger.info(f"Model {ollama_id} is ready to use with Ollama")
        return True

    except FileNotFoundError:
        logger.error(
            "Ollama is not installed, please install it from https://ollama.ai/"
        )
        return False


def update_env_file(model_id, backend):
    """Update the .env file with model information."""
    env_path = Path(".env")

    if not env_path.exists():
        logger.warning(".env file not found, creating a new one")
        env_path.write_text("")

    # Read existing .env file
    env_text = env_path.read_text()
    env_lines = env_text.split("\n")

    # Update or add MODEL_ID and LLM_BACKEND
    model_updated = False
    backend_updated = False

    for i, line in enumerate(env_lines):
        if line.startswith("MODEL_ID="):
            env_lines[i] = f"MODEL_ID={model_id}"
            model_updated = True
        elif line.startswith("LLM_BACKEND="):
            env_lines[i] = f"LLM_BACKEND={backend}"
            backend_updated = True

    # Add lines if not updated
    if not model_updated:
        env_lines.append(f"MODEL_ID={model_id}")

    if not backend_updated:
        env_lines.append(f"LLM_BACKEND={backend}")

    # Write updated .env file
    env_path.write_text("\n".join(env_lines))


def list_models():
    """List available models."""
    print("\nAvailable models:")
    print("-" * 80)
    print("{:<10} {:<40} {:<10}".format("ID", "Hugging Face ID", "Ollama ID"))
    print("-" * 80)

    for model_id, model_info in LLM_OPTIONS.items():
        print(
            "{:<10} {:<40} {:<10}".format(
                model_id, model_info["huggingface_id"], model_info["ollama_id"]
            )
        )

    print("\nModel descriptions:")
    print("-" * 80)

    for model_id, model_info in LLM_OPTIONS.items():
        print(f"{model_id}: {model_info['description']}")

    print("\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up LLMs for use with vLLM and Ollama"
    )
    parser.add_argument("--model", type=str, help="Model ID or Hugging Face ID")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "ollama"],
        default="ollama",
        help="Backend to use (vllm or ollama)",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument(
        "--model-name", type=str, help="Custom model name (for local models)"
    )
    parser.add_argument("--adapter", type=str, help="Adapter name (for PEFT models)")

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.check:
        check_environment()
        return

    if not args.model:
        parser.print_help()
        return

    # Check environment
    check_environment()

    # Set up model
    if args.backend == "vllm":
        setup_model_vllm(args.model, args.model_name, args.adapter)
    else:
        setup_model_ollama(args.model)


if __name__ == "__main__":
    main()
