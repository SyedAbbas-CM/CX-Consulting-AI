#!/usr/bin/env python
"""
Embedding Model Download Script

This script downloads embedding models for offline use.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_embeddings")

# Define embedding models
EMBEDDING_MODELS = {
    "bge-small": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "description": "Small BGE model (specialized for RAG) - 30M parameters"
    },
    "bge-base": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "description": "Base BGE model (specialized for RAG) - 110M parameters"
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "description": "Large BGE model (specialized for RAG) - 330M parameters"
    },
    "all-minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Compact all-purpose model - 80M parameters"
    },
    "all-mpnet": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "description": "High performance all-purpose model - 110M parameters"
    }
}

def download_model(model_id, output_dir=None):
    """Download a model for offline use."""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Downloading model: {model_id}")
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cache_folder = output_dir
        else:
            cache_folder = None
        
        # Load model (this will download it if not already cached)
        model = SentenceTransformer(model_id, cache_folder=cache_folder)
        
        # Get model size
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Model downloaded: {model_id}")
        logger.info(f"Parameter count: {model_size:,}")
        
        if cache_folder:
            logger.info(f"Model saved to: {cache_folder}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def list_models():
    """List available models."""
    logger.info("Available embedding models:")
    
    for model_name, model_info in EMBEDDING_MODELS.items():
        logger.info(f"- {model_name}: {model_info['description']}")
        logger.info(f"  Model ID: {model_info['model_id']}")
        logger.info("")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download embedding models for offline use")
    parser.add_argument("--model", type=str, help="Model to download (use 'list' to see available models)")
    parser.add_argument("--output-dir", type=str, help="Directory to save the model")
    parser.add_argument("--all", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        parser.print_help()
        return
    
    if args.model == "list":
        list_models()
        return
    
    # Install sentence-transformers if not already installed
    try:
        import sentence_transformers
    except ImportError:
        logger.info("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    
    # Download specified model
    if args.all:
        logger.info("Downloading all models...")
        for model_name, model_info in EMBEDDING_MODELS.items():
            model_output_dir = os.path.join(args.output_dir, model_name) if args.output_dir else None
            download_model(model_info["model_id"], model_output_dir)
    else:
        if args.model in EMBEDDING_MODELS:
            model_id = EMBEDDING_MODELS[args.model]["model_id"]
            download_model(model_id, args.output_dir)
        else:
            # Try to download the model ID directly
            download_model(args.model, args.output_dir)

if __name__ == "__main__":
    main() 