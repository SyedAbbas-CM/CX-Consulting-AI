#!/usr/bin/env python3
"""
Download and Prepare LLM Model

This script downloads LLM models from Hugging Face and converts them to GGUF format
for use with llama.cpp.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cx_consulting_ai.download_llm")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

SUPPORTED_MODELS = {
    # 7B Models
    "llama2-7b": {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "filename": None,  # Download entire repo
        "description": "Llama 2 7B base model",
        "quantization": "Q4_K_M"
    },
    "llama2-7b-chat": {
        "repo_id": "meta-llama/Llama-2-7b-chat-hf",
        "filename": None,
        "description": "Llama 2 7B chat model",
        "quantization": "Q4_K_M"
    },
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-v0.1",
        "filename": None,
        "description": "Mistral 7B base model",
        "quantization": "Q4_K_M"
    },
    "mistral-7b-instruct": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "filename": None,
        "description": "Mistral 7B instruct model",
        "quantization": "Q4_K_M"
    },
    
    # 13-14B Models
    "llama2-13b": {
        "repo_id": "meta-llama/Llama-2-13b-hf",
        "filename": None,
        "description": "Llama 2 13B base model",
        "quantization": "Q4_K_M"
    },
    "llama2-13b-chat": {
        "repo_id": "meta-llama/Llama-2-13b-chat-hf",
        "filename": None,
        "description": "Llama 2 13B chat model",
        "quantization": "Q4_K_M"
    },
    "vicuna-13b": {
        "repo_id": "lmsys/vicuna-13b-v1.5",
        "filename": None,
        "description": "Vicuna 13B model",
        "quantization": "Q4_K_M"
    },
    
    # Already quantized models (direct download)
    "gemma-2b-it-gguf": {
        "repo_id": "lmstudio-community/gemma-2b-it-GGUF",
        "filename": "gemma-2b-it.Q4_K_M.gguf",
        "description": "Gemma 2B instruct model (quantized)",
        "direct_gguf": True
    },
    "mistral-7b-instruct-gguf": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral 7B instruct model (quantized)",
        "direct_gguf": True
    },
    "llama2-13b-chat-gguf": {
        "repo_id": "TheBloke/Llama-2-13B-chat-GGUF",
        "filename": "llama-2-13b-chat.Q4_K_M.gguf",
        "description": "Llama 2 13B chat model (quantized)",
        "direct_gguf": True
    }
}

def download_model(model_name: str, output_dir: str, force: bool = False) -> str:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
        force: Whether to overwrite existing files
        
    Returns:
        Path to the downloaded model
    """
    if model_name not in SUPPORTED_MODELS:
        logger.error(f"Model {model_name} not supported. Available models: {', '.join(SUPPORTED_MODELS.keys())}")
        return None
    
    model_info = SUPPORTED_MODELS[model_name]
    repo_id = model_info["repo_id"]
    output_path = os.path.join(output_dir, model_name)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        if model_info.get("direct_gguf", False):
            # This is a pre-quantized GGUF model, just download the file
            filename = model_info["filename"]
            logger.info(f"Downloading quantized GGUF model: {repo_id}/{filename}")
            
            gguf_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=output_path
            )
            
            # Copy to models directory
            target_path = os.path.join(output_dir, filename)
            if os.path.exists(target_path) and not force:
                logger.info(f"Model already exists at {target_path}")
            else:
                shutil.copy(gguf_path, target_path)
                logger.info(f"Copied model to {target_path}")
            
            return target_path
        else:
            # Download the entire model repository
            logger.info(f"Downloading model: {repo_id}")
            
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=output_path
            )
            
            logger.info(f"Model downloaded to {model_path}")
            return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return None

def convert_to_gguf(model_path: str, output_dir: str, model_name: str, quantization: str = "Q4_K_M") -> str:
    """
    Convert a model to GGUF format using llama.cpp.
    
    Args:
        model_path: Path to the downloaded model
        output_dir: Directory to save the GGUF model
        model_name: Name of the model
        quantization: Quantization method
        
    Returns:
        Path to the converted GGUF model
    """
    try:
        # Check if llama.cpp is installed
        llama_cpp_path = shutil.which("llama-convert")
        if not llama_cpp_path:
            logger.error("llama.cpp not found. Please install llama.cpp first.")
            return None
        
        # Output path for GGUF model
        gguf_filename = f"{model_name}.{quantization}.gguf"
        gguf_path = os.path.join(output_dir, gguf_filename)
        
        # Check if GGUF model already exists
        if os.path.exists(gguf_path):
            logger.info(f"GGUF model already exists at {gguf_path}")
            return gguf_path
        
        # Convert to GGUF
        logger.info(f"Converting model to GGUF format with quantization {quantization}")
        
        cmd = [
            "python", "-m", "llama_cpp.convert_hf_to_gguf",
            "--verbose",
            "--outfile", gguf_path,
            "--quantize", quantization,
            model_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error converting model: {result.stderr}")
            return None
        
        logger.info(f"Model converted to GGUF format: {gguf_path}")
        return gguf_path
    
    except Exception as e:
        logger.error(f"Error converting model to GGUF: {str(e)}")
        return None

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Download and prepare LLM models")
    
    # Create a mutually exclusive group for --list and --model
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List available models")
    group.add_argument("--model", type=str, help="Model to download")
    
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--convert", action="store_true", help="Convert to GGUF format")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    # List available models
    if args.list:
        print("Available models:")
        for name, info in SUPPORTED_MODELS.items():
            print(f"  {name}: {info['description']}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download model
    model_path = download_model(args.model, args.output_dir, args.force)
    if not model_path:
        return 1
    
    # Convert to GGUF if requested
    if args.convert and not SUPPORTED_MODELS[args.model].get("direct_gguf", False):
        gguf_path = convert_to_gguf(
            model_path, 
            args.output_dir, 
            args.model, 
            SUPPORTED_MODELS[args.model]["quantization"]
        )
        if not gguf_path:
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 