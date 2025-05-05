#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import time
import json
import logging
import random
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import psutil # For checking PID status
import signal # For checking PID status on Unix-like systems

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_manager")

# Load environment variables
load_dotenv()

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Available model configurations
# Updated list: Gemma 3 IT, Qwen3 Instruct, and Gemma 2B IT
AVAILABLE_MODELS = {
    # --- Gemma Models ---
    "gemma-2b-it": {
        "repo_id": "google/gemma-2b-it",
        "filename": "gemma-2b-it.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf",
        "size_gb": 1.4,
        "description": "Gemma 2B Instruct (Q4_K_M)",
        "type": "instruct"
    },
    "gemma-3-4b-it": { 
        "repo_id": "google/gemma-3-4b-it",
        "filename": "gemma-3-4b-it-im_q4_k_m.gguf", # Common GGUF naming
        # Corrected URL using lmstudio-community repo (assuming Q4_K_M exists, adjust filename if needed)
        "url": "https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-im-q4_k_m.gguf", 
        "size_gb": 2.8, # Estimate
        "description": "Gemma 3 4B Instruct (Q4_K_M)",
        "type": "instruct"
    },
    # "gemma-3-12b-it": { # Using lmstudio-community QAT GGUF
    #     "repo_id": "google/gemma-3-12b-it",
    #     "filename": "gemma-3-12b-it-qat-q4_0.gguf", # Using the Q4_0 filename from the repo
    #     "url": "https://huggingface.co/lmstudio-community/gemma-3-12B-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-q4_0.gguf", 
    #     "size_gb": 6.9, # Actual size from repo
    #     "description": "Gemma 3 12B Instruct QAT (Q4_0)",
    #     "type": "instruct"
    # }, 
    "gemma-3-27b-it": { # Using lmstudio-community QAT GGUF
        "repo_id": "google/gemma-3-27b-it",
        "filename": "gemma-3-27b-it-qat-q4_0.gguf", # Using the Q4_0 filename from the repo
        "url": "https://huggingface.co/lmstudio-community/gemma-3-27B-it-qat-GGUF/resolve/main/gemma-3-27b-it-qat-q4_0.gguf",
        "size_gb": 15.6, # Actual size from repo
        "description": "Gemma 3 27B Instruct QAT (Q4_0)",
        "type": "instruct"
    },
    # --- Qwen3 Models --- 
    "qwen3-4b-instruct": {
        "repo_id": "Qwen/Qwen3-4B-Instruct", 
        "filename": "qwen3-4b-instruct-q4_k_m.gguf",
        # Corrected URL using lmstudio-community repo
        "url": "https://huggingface.co/lmstudio-community/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        "size_gb": 2.5, # Actual size from repo
        "description": "Qwen3 4B Instruct (Q4_K_M)",
        "type": "instruct"
    },
     "qwen3-14b-instruct": {
        "repo_id": "Qwen/Qwen3-14B-Instruct", 
        "filename": "qwen3-14b-instruct-q4_k_m.gguf",
        # Corrected URL using lmstudio-community repo
        "url": "https://huggingface.co/lmstudio-community/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
        "size_gb": 9.0, # Actual size from repo
        "description": "Qwen3 14B Instruct (Q4_K_M)",
        "type": "instruct"
    },
     "qwen3-32b-instruct": {
        "repo_id": "Qwen/Qwen3-32B-Instruct", 
        "filename": "qwen3-32b-instruct-q4_k_m.gguf",
        # Corrected URL using lmstudio-community repo
        "url": "https://huggingface.co/lmstudio-community/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf",
        "size_gb": 19.8, # Actual size from repo
        "description": "Qwen3 32B Instruct (Q4_K_M)",
        "type": "instruct"
    }
}

# Model benchmark test cases
CX_BENCHMARK_TESTS = [
    {
        "name": "cx_strategy",
        "prompt": """You are a professional CX (Customer Experience) consultant assistant. 
        
Create a CX strategy outline for a retail company that wants to improve customer satisfaction scores.
Include specific initiatives across multiple touchpoints and how success would be measured.

Your strategy:""",
        "max_tokens": 1024,
        "temperature": 0.3
    },
    {
        "name": "journey_map", 
        "prompt": """You are a professional CX (Customer Experience) consultant assistant.

Create a customer journey map for a first-time user of a mobile banking app, from download to completing their first transaction.
Include the stages, customer emotions, pain points, and opportunities for improvement at each stage.

Your journey map:""",
        "max_tokens": 1024,
        "temperature": 0.3
    },
    {
        "name": "roi_analysis",
        "prompt": """You are a professional CX (Customer Experience) consultant assistant.

Create an ROI analysis for implementing a new customer feedback system for an e-commerce company.
Current metrics: NPS of 32, customer retention rate of 65%, average order value of $85
Cost of implementation: $150,000

Your ROI analysis:""",
        "max_tokens": 1024,
        "temperature": 0.3
    }
]

# Model template formats
MODEL_TEMPLATES = {
    "instruct": {
        "gemma": """<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
""",
        "mistral": """
{prompt} 
""",
        "llama": """
{prompt} 
""",
        "phi": """<|user|>
{prompt}
<|assistant|>""",
        "qwen": """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    },
    "chat": {
        "gemma": """<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
""",
        "mistral": """
{prompt} 
""",
        "llama": """
{prompt} 
""",
        "phi": """<|user|>
{prompt}
<|assistant|>""",
        "qwen": """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    }
}

# At the top of the file, after imports but before other code
# Add this global variable
SUPPRESS_RESUME_PROMPT = False
MAX_RETRIES = 10  # Maximum number of retries for download failures

# Available mirror configurations
MIRROR_CONFIGS = {
    "hf": {
        "name": "Hugging Face",
        "description": "Primary source - Hugging Face model repository",
        "url_pattern": "{url}"  # Just use the original URL
    },
    "hf-cdn": {
        "name": "Hugging Face CDN",
        "description": "Hugging Face CDN - might be faster in some regions",
        "url_pattern": "{url}".replace("resolve/main", "resolve/main")  # Same as original for now
    }
}

DEFAULT_MIRROR = "hf"  # Default mirror to use

# --- Lock File Management ---

def _get_lock_file_path(model_filename: str) -> Path:
    """Gets the expected path for a model's lock file."""
    return Path(MODELS_DIR) / f"{model_filename}.lock"

def _create_lock_file(model_filename: str) -> bool:
    """Creates a lock file for the model download process."""
    lock_file = _get_lock_file_path(model_filename)
    try:
        pid = os.getpid()
        lock_data = {
            "pid": pid,
            "start_time": time.time(),
            "status": "downloading"
        }
        with open(lock_file, 'w') as f:
            json.dump(lock_data, f)
        logger.info(f"Created lock file: {lock_file} for PID {pid}")
        return True
    except IOError as e:
        logger.error(f"Failed to create lock file {lock_file}: {e}")
        return False

def _read_lock_file(model_filename: str) -> Optional[Dict]:
    """Reads data from a lock file."""
    lock_file = _get_lock_file_path(model_filename)
    if not lock_file.exists():
        return None
    try:
        with open(lock_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read or parse lock file {lock_file}: {e}")
        return None # Consider it invalid

def _remove_lock_file(model_filename: str):
    """Removes the lock file."""
    lock_file = _get_lock_file_path(model_filename)
    try:
        if lock_file.exists():
            lock_file.unlink()
            logger.info(f"Removed lock file: {lock_file}")
    except IOError as e:
        logger.error(f"Failed to remove lock file {lock_file}: {e}")

def _is_pid_running(pid: int) -> bool:
    """Checks if a process with the given PID is running."""
    if pid <= 0:
        return False
    if 'psutil' in sys.modules:
        try:
            return psutil.pid_exists(pid)
        except Exception as e:
            logger.warning(f"psutil check failed for PID {pid}: {e}. Falling back.")

    if os.name == 'posix':
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False 
    elif os.name == 'nt':
        try:
            output = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}'], stderr=subprocess.STDOUT)
            return str(pid) in output.decode()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    else:
        logger.warning(f"PID running check not implemented for OS: {os.name}")
        return False 

# --- End Lock File Management ---

def get_prompt_template(model_id):
    """Get the appropriate prompt template for a model"""
    if model_id not in AVAILABLE_MODELS:
        # Default to a generic template
        return "{prompt}"
    
    model_info = AVAILABLE_MODELS[model_id]
    model_type = model_info["type"]
    
    # Determine model family from filename or repo_id
    filename = model_info["filename"].lower()
    repo_id = model_info["repo_id"].lower()
    
    if "gemma" in filename or "gemma" in repo_id:
        family = "gemma"
    elif "mistral" in filename or "mistral" in repo_id or "mixtral" in filename or "mixtral" in repo_id:
        family = "mistral"
    elif "llama" in filename or "llama" in repo_id:
        family = "llama"
    elif "phi" in filename or "phi" in repo_id:
        family = "phi"
    elif "qwen" in filename or "qwen" in repo_id:
        family = "qwen"
    else:
        # Default to a generic template
        return "{prompt}"
    
    # Get template for the model family and type
    if family in MODEL_TEMPLATES[model_type]:
        return MODEL_TEMPLATES[model_type][family]
    else:
        # Default to a generic template
        return "{prompt}"


def download_with_retry(url, temp_path, headers, total_size, resume_size, model_id):
    """Downloads a file with retries and progress bar, including authentication."""
    retries = 0
    # Add HF Token if available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        logger.info("Using Hugging Face token for authentication.")
    else:
        logger.warning("HF_TOKEN environment variable not set. Download might fail for restricted models.")

    while retries < MAX_RETRIES:
        try:
            # Update headers with current file size before each attempt
            if resume_size > 0:
                headers['Range'] = f'bytes={resume_size}-'
                logger.info(f"Resuming download from byte position {resume_size} ({resume_size/(1024*1024):.2f} MB)")
            else:
                # Remove Range header if starting fresh
                if 'Range' in headers:
                    headers.pop('Range')
                
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            
            # If server doesn't support resume, we need to start over
            if resume_size > 0 and response.status_code != 206 and 'Range' in headers:
                logger.warning("Server doesn't support resuming downloads. Starting fresh.")
                resume_size = 0
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                # Try again without range header
                headers.pop('Range', None)
                response = requests.get(url, stream=True, headers=headers)
                response.raise_for_status()
            
            # For resumed downloads with 206 Partial Content, adjust total size calculation
            adjusted_total = total_size
            if resume_size > 0 and response.status_code == 206:
                remaining_size = int(response.headers.get('content-length', 0))
                adjusted_total = resume_size + remaining_size
                
            # Initialize tqdm with the total size
            with tqdm(total=adjusted_total, initial=resume_size, unit='B', unit_scale=True, desc=model_id) as pbar:
                # Open the output file in append mode if resuming, otherwise write mode
                file_mode = 'ab' if resume_size > 0 else 'wb'
                with open(temp_path, file_mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            # Ensure we flush to disk regularly to avoid data loss on crash
                            if random.randint(0, 100) < 5:  # ~5% chance to flush
                                f.flush()
                                os.fsync(f.fileno())
                                
            return True
            
        except requests.exceptions.RequestException as e:
            # Save the current position before retry
            if os.path.exists(temp_path):
                resume_size = os.path.getsize(temp_path)
                logger.info(f"Download interrupted at {resume_size/(1024*1024):.2f} MB")
                # Ensure the file is properly flushed
                try:
                    with open(temp_path, 'ab') as f:
                        f.flush()
                        os.fsync(f.fileno())
                except:
                    pass
            
            retries += 1
            retry_wait = min(30, 2 ** retries) + random.random()  # Exponential backoff with jitter, max 30 seconds
            
            if retries < MAX_RETRIES:
                logger.warning(f"Download attempt {retries} failed: {str(e)}")
                logger.info(f"Retrying in {retry_wait:.1f} seconds from position {resume_size/(1024*1024):.2f} MB...")
                time.sleep(retry_wait)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                logger.info(f"Partial download saved at {temp_path} ({resume_size/(1024*1024):.2f} MB)")
                return False
                
        except KeyboardInterrupt:
            if os.path.exists(temp_path):
                resume_size = os.path.getsize(temp_path)
                logger.info(f"Download interrupted by user at {resume_size/(1024*1024):.2f} MB")
                logger.info(f"Partial download saved at {temp_path}")
                logger.info("You can resume the download later with --resume option")
            raise
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            if os.path.exists(temp_path):
                resume_size = os.path.getsize(temp_path)
                logger.info(f"Partial download saved at {temp_path} ({resume_size/(1024*1024):.2f} MB)")
            return False
            
    return False  # If we get here, all retries failed


def get_download_url(model_id, mirror=DEFAULT_MIRROR):
    """Gets the download URL for a model, potentially using a mirror."""
    if model_id not in AVAILABLE_MODELS:
        logger.error(f"Model ID '{model_id}' not found in available models.")
        return None

    model_info = AVAILABLE_MODELS[model_id]
    original_url = model_info.get("url")

    if not original_url:
        logger.error(f"URL not defined for model ID '{model_id}'.")
        return None

    # Use the specified mirror configuration
    if mirror not in MIRROR_CONFIGS:
        logger.warning(f"Mirror '{mirror}' not found. Using default mirror '{DEFAULT_MIRROR}'.")
        mirror = DEFAULT_MIRROR

    mirror_config = MIRROR_CONFIGS[mirror]
    url_pattern = mirror_config.get("url_pattern")

    if not url_pattern:
        logger.warning(f"URL pattern not defined for mirror '{mirror}'. Using original URL.")
        return original_url

    try:
        # Format the URL using the pattern and original URL
        # Ensure necessary components are present in the original URL if the pattern needs them
        # Example assumes pattern just uses the original URL directly or slightly modified
        download_url = url_pattern.format(url=original_url) # Potential KeyError if pattern needs other keys
        return download_url
    except KeyError as e:
        logger.error(f"Error formatting URL with pattern for mirror '{mirror}': Missing key {e}. Using original URL.")
        return original_url
    except Exception as e: # Added generic except block to satisfy L454
        logger.error(f"Unexpected error getting download URL for mirror '{mirror}': {e}. Using original URL.")
        return original_url


def download_model(model_id, force=False, mirror=DEFAULT_MIRROR):
    """Downloads the specified model if not already present or if force is True."""
    model_info = AVAILABLE_MODELS.get(model_id)
    if not model_info:
        logger.error(f"Model ID '{model_id}' not found in AVAILABLE_MODELS.")
        return False
    model_filename = model_info["filename"]
    model_path = Path(MODELS_DIR) / model_filename
    temp_path = Path(MODELS_DIR) / f"{model_filename}.part"
    lock_file_path = _get_lock_file_path(model_filename)

    # Check for existing lock file
    if lock_file_path.exists():
        lock_data = _read_lock_file(model_filename)
        if lock_data:
            pid = lock_data.get('pid')
            if pid and _is_pid_running(pid):
                logger.warning(f"Download for '{model_filename}' already in progress (PID: {pid}). Aborting.")
                return False # Indicate download did not proceed
            else:
                logger.warning(f"Found stale lock file for '{model_filename}'. Removing it.")
                _remove_lock_file(model_filename)
        else:
            # Lock file exists but is invalid/unreadable
            logger.warning(f"Found invalid lock file for '{model_filename}'. Removing it.")
            _remove_lock_file(model_filename)

    model_path = Path(MODELS_DIR) / model_filename
    temp_path = Path(MODELS_DIR) / f"{model_filename}.part"

    if os.path.exists(model_path) and not force:
        logger.info(f"Model '{model_filename}' already exists. Skipping download.")
        return True

    if os.path.exists(model_path) and force:
        logger.info(f"Force download specified. Removing existing model '{model_filename}'.")
        try:
            os.remove(model_path)
        except OSError as e:
            logger.error(f"Error removing existing model '{model_path}': {e}")
            return False # Stop if we can't remove the old one

    resume_size = 0
    if os.path.exists(temp_path):
        resume_size = os.path.getsize(temp_path)
        logger.info(f"Resuming download for '{model_filename}' from {resume_size} bytes.")

    download_url = get_download_url(model_id, mirror=mirror)
    if not download_url:
        return False # Error logged in get_download_url

    # Create lock file before starting download
    if not _create_lock_file(model_filename):
        logger.error(f"Failed to create lock file for '{model_filename}'. Aborting download.")
        return False

    try:
        logger.info(f"Starting download: {model_id} from {download_url}")
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        # Correct resume handling based on response headers/status if server supports it
        # For simplicity, just check if total_size matches expected or is zero

        headers = {}
        if resume_size > 0:
            headers['Range'] = f'bytes={resume_size}-'
            # Re-request with range header
            response = requests.get(download_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            # Check 206 Partial Content status if resuming
            if response.status_code != 206:
                logger.warning(f"Server did not support resume (Status: {response.status_code}). Restarting download.")
                resume_size = 0
                # Re-request without range
                response = requests.get(download_url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
            else:
                 # Adjust total size calculation if needed based on Content-Range header
                 pass # Assuming total_size from initial request is sufficient

        # Use tqdm for progress bar
        progress_bar = tqdm(
            total=total_size,
            initial=resume_size,
            unit='iB',
            unit_scale=True,
            desc=model_filename
        )

        # Open file in append binary mode ('ab') if resuming, write binary ('wb') otherwise
        file_mode = 'ab' if resume_size > 0 else 'wb'
        with open(temp_path, file_mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            logger.error(f"Download failed: Expected {total_size} bytes, got {progress_bar.n} bytes")
            # Optionally remove temp file here
            # try: os.remove(temp_path) except OSError: pass
            return False
        else:
             # Download finished, rename temp file to final name
             os.rename(temp_path, model_path)
             logger.info(f"Model '{model_filename}' downloaded successfully.")
             return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Download error for {model_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download: {e}", exc_info=True)
        return False
    finally:
        # Always remove lock file when download attempt finishes (success or fail)
        _remove_lock_file(model_filename)


def benchmark_model(model_path, test_cases=None, num_runs=1, model_id=None):
    """Benchmark a model using the llama.cpp backend"""
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist")
        return {}
    
    if test_cases is None:
        test_cases = CX_BENCHMARK_TESTS
    
    # Determine the model ID if not provided
    if model_id is None:
        model_filename = os.path.basename(model_path)
        # Try to find the model ID from the filename
        for mid, info in AVAILABLE_MODELS.items():
            if info["filename"] == model_filename:
                model_id = mid
                break
    
    # Get the appropriate prompt template if model_id is available
    prompt_template = get_prompt_template(model_id) if model_id else "{prompt}"
    
    # Set up llama.cpp backend
    try:
        from llama_cpp import Llama
        
        logger.info(f"Loading model from {model_path}...")
        model = Llama(
            model_path=model_path,
            n_ctx=8192,  # Context size
            n_gpu_layers=-1  # Use all available GPU layers
        )
        
        # Run benchmark tests
        results = {}
        
        for test in test_cases:
            test_name = test["name"]
            logger.info(f"Running benchmark test: {test_name}")
            
            # Format the prompt with the template
            formatted_prompt = prompt_template.format(prompt=test["prompt"])
            
            times = []
            tokens_per_second = []
            outputs = []
            
            for i in range(num_runs):
                start_time = time.time()
                
                output = model(
                    formatted_prompt,
                    max_tokens=test["max_tokens"],
                    temperature=test["temperature"],
                    stop=["User:", "\n\n\n"],
                    echo=False
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                if output and "choices" in output and len(output["choices"]) > 0:
                    text = output["choices"][0]["text"]
                    tokens = len(model.tokenize(text.encode()))
                    tokens_sec = tokens / elapsed if elapsed > 0 else 0
                    
                    times.append(elapsed)
                    tokens_per_second.append(tokens_sec)
                    outputs.append(text)
                
                logger.info(f"  Run {i+1}: {elapsed:.2f}s, {tokens_sec:.2f} tokens/sec")
            
            # Calculate average metrics
            avg_time = sum(times) / len(times) if times else 0
            avg_tokens_sec = sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0
            
            results[test_name] = {
                "average_time": avg_time,
                "average_tokens_per_second": avg_tokens_sec,
                "outputs": outputs
            }
            
            logger.info(f"Completed test {test_name}: avg {avg_time:.2f}s, {avg_tokens_sec:.2f} tokens/sec")
        
        return results
    
    except ImportError:
        logger.error("llama-cpp-python is not installed. Please install it with:")
        logger.error("pip install llama-cpp-python")
        return {}
    except Exception as e:
        logger.error(f"Error benchmarking model: {str(e)}")
        return {}


def update_env_config(model_id):
    """Update the .env file with the selected model's MODEL_PATH."""
    if model_id not in AVAILABLE_MODELS:
        logger.error(f"Model '{model_id}' not found in available models")
        return False
    
    model_info = AVAILABLE_MODELS[model_id]
    # Use the parent directory's .env file relative to the script's location
    # Script: app/scripts/model_manager.py -> Parent: app/ -> Grandparent: ./ (workspace root)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    model_filename = model_info["filename"]
    model_path_value = f"models/{model_filename}" # Use relative path
    
    if not os.path.exists(env_path):
        logger.warning(f".env file not found at expected location: {env_path}")
        # Optionally create a minimal .env file
        try:
            with open(env_path, 'w') as f:
                f.write(f"MODEL_PATH={model_path_value}\n")
                f.write(f"LLM_BACKEND=llama.cpp\n") # Add backend setting too
            logger.info(f"Created new .env file at {env_path} with model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating .env file: {str(e)}")
            return False
    
    try:
        # Read the current .env file
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update or add the MODEL_PATH line
        updated_lines = []
        model_path_set = False
        llm_backend_set = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("MODEL_PATH="):
                updated_lines.append(f"MODEL_PATH={model_path_value}\n")
                model_path_set = True
            elif stripped_line.startswith("# MODEL_PATH="): # Handle commented out line
                 updated_lines.append(f"MODEL_PATH={model_path_value}\n") # Uncomment and set
                 model_path_set = True
            elif stripped_line.startswith("MODEL_ID="): # Comment out MODEL_ID if MODEL_PATH is set
                updated_lines.append(f"# {stripped_line}\n") 
            elif stripped_line.startswith("LLM_BACKEND="):
                 updated_lines.append(line) # Keep existing backend setting
                 llm_backend_set = True
            else:
                updated_lines.append(line)
        
        # Add MODEL_PATH if it wasn't found
        if not model_path_set:
            updated_lines.append(f"MODEL_PATH={model_path_value}\n")
        
        # Ensure LLM_BACKEND is set to llama.cpp if not present
        if not llm_backend_set:
             updated_lines.append(f"LLM_BACKEND=llama.cpp\n")

        # Write the updated content back to the .env file
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        logger.info(f"Updated {env_path} to use MODEL_PATH={model_path_value} for model {model_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating .env file: {str(e)}")
        return False


def print_available_models():
    """Print the list of available models"""
    print("\nAvailable Models:")
    print("-" * 80)
    print(f"{'ID':<15} {'Size':<10} {'Type':<10} Description")
    print("-" * 80)
    
    for model_id, info in AVAILABLE_MODELS.items():
        print(f"{model_id:<15} {info['size_gb']:<10.1f}GB {info['type']:<10} {info['description']}")
    print()


def check_current_model():
    """Check which model is currently configured via MODEL_PATH in the .env file"""
    # Check the root .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    
    current_model_path_value = None
    model_id = None
    model_filename = None
    config_file_path = None
    
    if os.path.exists(env_path):
        config_file_path = env_path
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line.startswith("MODEL_PATH=") and not stripped_line.startswith("#"):
                        # Get the path value, remove potential quotes
                        current_model_path_value = stripped_line.split("=", 1)[1].strip('\"''') 
                        # Extract filename from the path
                        if "/" in current_model_path_value:
                            model_filename = current_model_path_value.split("/")[-1]
                        else:
                            model_filename = current_model_path_value # Path might be relative like models/model.gguf
                        
                        # Find corresponding model_id from filename
                        for mid, info in AVAILABLE_MODELS.items():
                            if info.get("filename") == model_filename:
                                model_id = mid
                                break # Found the matching model ID
                        break # Found MODEL_PATH line
        except Exception as e:
            logger.error(f"Error reading {env_path}: {str(e)}")
            config_file_path = None # Indicate error reading config

    return model_id, model_filename, config_file_path # Return found info


def print_available_mirrors():
    """Print the list of available download mirrors"""
    print("\nAvailable Mirrors:")
    print("-" * 80)
    print(f"{'ID':<10} {'Name':<20} Description")
    print("-" * 80)
    
    for mirror_id, info in MIRROR_CONFIGS.items():
        print(f"{mirror_id:<10} {info['name']:<20} {info['description']}")
    print()


def get_available_models():
    """Return the list of available models."""
    return AVAILABLE_MODELS

# --- New Function: get_model_status ---
def get_model_status(model_id):
    """Check the status of a model file (available, downloading, download_failed, not_available, not_found)."""
    if model_id not in AVAILABLE_MODELS:
        return {"status": "not_found", "message": f"Model ID '{model_id}' not defined."}
    
    model_info = AVAILABLE_MODELS[model_id]
    model_filename = model_info["filename"]
    model_path = Path(MODELS_DIR) / model_filename
    temp_path = Path(MODELS_DIR) / f"{model_filename}.part"
    lock_file_path = _get_lock_file_path(model_filename)

    # 1. Check lock file first
    lock_data = _read_lock_file(model_filename)
    if lock_data:
        pid = lock_data.get("pid")
        if pid and _is_pid_running(pid):
            # Process is actively running
            elapsed_time = time.time() - lock_data.get("start_time", time.time())
            return {"status": "downloading", "message": f"Download in progress (PID: {pid}, Running for {elapsed_time:.0f}s)", "pid": pid}
        else:
            # Lock file exists, but PID is invalid or not running (Stale lock)
            # FIX: Log the warning correctly here, using the pid we have
            pid_info = f"PID {pid}" if pid else "Invalid PID"
            logger.warning(f"Found stale lock file for model {model_id} ({pid_info} not running). Consider download failed.")
            # Option: Automatically remove stale lock here?
            # _remove_lock_file(model_filename) 
            # Report as failed. If a .part file also exists, it reinforces failure.
            status_msg = f"Previous download process ({pid_info}) is not running (stale lock)."
            if temp_path.exists():
                 status_msg += " Partial file found."
                 return {"status": "download_failed", "message": status_msg, "path": str(temp_path)} 
            else:
                 # Stale lock but no .part or final file? Odd state, but treat as failed/not available.
                 return {"status": "download_failed", "message": status_msg + " No partial file found."}
    # --- End Lock File Check --- If lock_data was None, we continue below ---
    
    # --- No valid lock file found, proceed with file checks --- 

    # 2. Check final file
    if model_path.exists():
        # Optional: Add file size verification here
        return {"status": "available", "message": "Model is downloaded and available.", "path": str(model_path)}

    # 3. Check partial file (implies failed/interrupted download because no lock file)
    if temp_path.exists():
        return {"status": "download_failed", "message": "Partial download file exists, but download is not active (no lock file).", "path": str(temp_path)}

    # 4. Not found / Not available (no lock, no final, no partial)
    return {"status": "not_available", "message": "Model is not downloaded."}
# --- End New Function ---

def main():
    parser = argparse.ArgumentParser(description="Manage and benchmark LLMs.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models and their status.")

    # List mirrors command
    list_mirrors_parser = subparsers.add_parser("list-mirrors", help="List available download mirrors.")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of a specific model.")
    status_parser.add_argument("model_id", help="ID of the model to check (e.g., gemma-2b-it)")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model.")
    download_parser.add_argument("model_id", help="ID of the model to download (e.g., gemma-2b-it)")
    download_parser.add_argument("--force", action="store_true", help="Force download even if model exists.")
    download_parser.add_argument("--mirror", default=DEFAULT_MIRROR, choices=MIRROR_CONFIGS.keys(), help="Mirror to use for download.")

    # Set Active command
    set_active_parser = subparsers.add_parser("set-active", help="Set a downloaded model as active in .env.")
    set_active_parser.add_argument("model_id", help="ID of the model to set active.")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark the currently active model.")
    benchmark_parser.add_argument("--runs", type=int, default=1, help="Number of times to run each test case.")
    benchmark_parser.add_argument("--model-id", help="Benchmark a specific model ID (must be downloaded).") # Allow overriding active

    args = parser.parse_args()

    if args.command == "list":
        print_available_models()
    elif args.command == "list-mirrors":
        print_available_mirrors()
    elif args.command == "status":
        status_info = get_model_status(args.model_id)
        print(json.dumps(status_info, indent=2))
    elif args.command == "download":
        success = download_model(args.model_id, force=args.force, mirror=args.mirror)
        if not success:
            sys.exit(1) # Exit with error code if download failed
    elif args.command == "set-active":
        # First, check if the model is actually available/downloaded
        status_info = get_model_status(args.model_id)
        if status_info["status"] != 'available':
            logger.error(f"Model '{args.model_id}' is not available (Status: {status_info['status']}). Cannot set active.")
            sys.exit(1)
        # Corrected: Use update_env_config correctly
        if update_env_config(model_id=args.model_id):
            logger.info(f"Model '{args.model_id}' set as active in .env. Restart application to apply changes.")
        else:
            logger.error(f"Failed to update .env for model '{args.model_id}'.")
            sys.exit(1)
    elif args.command == "benchmark":
        model_to_benchmark = args.model_id
        model_path_to_benchmark = None

        if model_to_benchmark:
             # Benchmark specific model ID
             if model_to_benchmark not in AVAILABLE_MODELS:
                 logger.error(f"Model ID '{model_to_benchmark}' not found.")
                 sys.exit(1)
             status_info = get_model_status(model_to_benchmark)
             if status_info["status"] != 'available':
                  logger.error(f"Model '{model_to_benchmark}' is not available (Status: {status_info['status']}). Cannot benchmark.")
                  sys.exit(1)
             model_filename = AVAILABLE_MODELS[model_to_benchmark]["filename"]
             model_path_to_benchmark = os.path.join(MODELS_DIR, model_filename)
        else:
             # Benchmark currently active model from .env
             active_model_id, active_filename, _ = check_current_model()
             if not active_model_id or not active_filename:
                  logger.error("Could not determine active model from .env. Specify --model-id or run set-active first.")
                  sys.exit(1)
             model_to_benchmark = active_model_id # Use the active model ID for logging/template
             model_path_to_benchmark = os.path.join(MODELS_DIR, active_filename)
             if not os.path.exists(model_path_to_benchmark):
                 logger.error(f"Active model file not found: {model_path_to_benchmark}")
                 sys.exit(1)

        logger.info(f"Benchmarking model: {model_to_benchmark} ({model_path_to_benchmark}) with {args.runs} run(s) per test.")
        results = benchmark_model(model_path_to_benchmark, test_cases=CX_BENCHMARK_TESTS, num_runs=args.runs, model_id=model_to_benchmark)
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))

    else:
        parser.print_help()

if __name__ == "__main__":
    main() 