import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tiktoken
from dotenv import find_dotenv, load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.config import Settings, get_settings
from app.scripts.model_manager import AVAILABLE_MODELS, MODELS_DIR

# Configure logger
logger = logging.getLogger("cx_consulting_ai.llm_service")

# Load environment variables
load_dotenv()

# Get settings - do this once at module level, or inside init if needed per-instance
settings = get_settings()

# Check which backend we're using
LLM_BACKEND = settings.LLM_BACKEND

if LLM_BACKEND == "vllm":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
elif LLM_BACKEND == "ollama":
    import requests
    from transformers import AutoTokenizer
elif LLM_BACKEND == "azure":
    try:
        import openai
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: openai package is not installed. Please install it with:")
        print("pip install openai>=1.0.0")
        sys.exit(1)
elif LLM_BACKEND == "llama.cpp":
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python is not installed. Please install it with:")
        print("pip install llama-cpp-python")
        sys.exit(1)


class LLMService:
    """High-performance LLM service with multiple backend options."""

    def __init__(self, gpu_count: int = None, max_model_len: int = None):
        """
        Initialize the LLM service configuration.
        Reads config directly from environment/settings.

        Args:
            gpu_count: Initial GPU count (can be overridden by env)
            max_model_len: Initial max sequence length (can be overridden by env)
        """
        # Get settings instance for this service instance
        self.settings = get_settings()

        # Store initial values for GPU/length, but not model path/id
        self._initial_gpu_count = gpu_count
        self._initial_max_model_len = max_model_len

        self.llm = None
        self.tokenizer = None
        self.client = None  # For Azure
        self.backend = None
        self.model_id = None
        self.model_path = None
        self.gpu_count = None
        self.max_model_len = None
        self.azure_endpoint = None
        self.azure_api_key = None
        self.azure_deployment = None
        self.ollama_base_url = None
        # Use N_THREADS from settings for the executor
        self.n_threads = self.settings.N_THREADS or os.cpu_count()
        self.chat_format = self.settings.CHAT_FORMAT

        # Create dedicated executor for LLM tasks (P2 / G1 / G2)
        self.llm_executor = ThreadPoolExecutor(
            max_workers=self.n_threads, thread_name_prefix="llm_worker"
        )
        logger.info(
            f"Initialized LLM ThreadPoolExecutor with {self.n_threads} workers."
        )

        self._update_config_from_env()  # Load initial config from .env
        self._load_model()  # Load the model based on initial config

    def _update_config_from_env(self):
        """Load or reload configuration from environment variables."""
        logger.info("Updating LLMService configuration from environment variables...")
        # Force reload of .env file
        env_path = find_dotenv(usecwd=True)  # Find .env in CWD or parent dirs
        if not env_path:
            logger.warning(".env file not found when updating LLMService config.")
            # Try loading without path specification as a fallback
            load_dotenv(override=True)
        else:
            logger.info(f"Loading environment variables from: {env_path}")
            load_dotenv(dotenv_path=env_path, override=True)

        # Reload settings object after loading .env
        settings = get_settings()

        self.backend = settings.LLM_BACKEND  # Update backend first
        self.model_id = settings.MODEL_ID
        self.model_path = settings.MODEL_PATH
        self.gpu_count = self._initial_gpu_count or settings.GPU_COUNT
        self.max_model_len = self._initial_max_model_len or settings.MAX_MODEL_LEN

        # Azure OpenAI specific settings
        self.azure_endpoint = settings.AZURE_OPENAI_ENDPOINT
        self.azure_api_key = settings.AZURE_OPENAI_KEY
        self.azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT

        # Ollama specific settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(
            f"LLMService config updated: backend={self.backend}, model_path={self.model_path}, model_id={self.model_id}, max_len={self.max_model_len}, gpu_count={self.gpu_count}"
        )

    def _load_model(self):
        """Load the LLM based on the current configuration."""
        # Clear existing model/client first to release resources (if any)
        if self.llm:
            logger.info(f"Unloading previous {self.backend} model...")
            # Specific cleanup might be needed depending on the backend (e.g., closing connections)
            if self.backend == "llama.cpp" and hasattr(
                self.llm, "close"
            ):  # Check if close method exists
                try:
                    # llama_cpp doesn't have an explicit close/unload in newer versions?
                    # del self.llm might be enough if GC works properly
                    pass
                except Exception as e:
                    logger.warning(
                        f"Could not explicitly close previous llama.cpp instance: {e}"
                    )
            self.llm = None
            self.tokenizer = None
            logger.info("Previous model unloaded.")
        if self.client:  # For Azure
            self.client = None  # Assuming the client doesn't need explicit closing

        logger.info(f"Loading model with backend: {self.backend}")
        logger.info(f"Using MODEL_PATH='{self.model_path}', MODEL_ID='{self.model_id}'")

        try:
            if self.backend == "vllm":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.llm = LLM(
                    model=self.model_id,
                    tensor_parallel_size=self.gpu_count,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=0.85,
                )
                logger.info(
                    f"Model '{self.model_id}' loaded successfully with vLLM backend"
                )

            elif self.backend == "ollama":
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                except Exception:
                    logger.warning(
                        f"Could not load tokenizer for '{self.model_id}', using gpt2 fallback."
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                # No LLM object to load for Ollama, just check connectivity?
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags")
                    response.raise_for_status()
                    logger.info(f"Ollama backend confirmed at {self.ollama_base_url}")
                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Failed to connect to Ollama backend at {self.ollama_base_url}: {e}"
                    )
                    # Decide if this should raise an error or just warn
                    # raise ConnectionError(f"Failed to connect to Ollama: {e}") from e

            elif self.backend == "azure":
                self.client = openai.AzureOpenAI(
                    api_key=self.azure_api_key,
                    api_version=os.getenv(
                        "AZURE_OPENAI_API_VERSION", "2023-12-01-preview"
                    ),
                    azure_endpoint=self.azure_endpoint,
                )
                logger.info(
                    f"Azure OpenAI client initialized for deployment: {self.azure_deployment}"
                )
                try:
                    # Try loading a tokenizer appropriate for the likely model type
                    # This is heuristic, might need adjustment based on common Azure models
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "gpt2"
                    )  # Fallback/common tokenizer
                except Exception:
                    self.tokenizer = None
                    logger.warning(
                        "Could not load gpt2 tokenizer for Azure token counting, will use estimation."
                    )

            elif self.backend == "llama.cpp":
                # Import necessary items here, only when backend is llama.cpp
                from llama_cpp import Llama

                model_file_to_load = None
                # Check 1: Use MODEL_PATH if it's set and points to an existing file
                if self.model_path and os.path.exists(self.model_path):
                    model_file_to_load = self.model_path
                    logger.info(
                        f"Using model file from MODEL_PATH: {model_file_to_load}"
                    )
                # Check 2: Use MODEL_ID if it's a direct path to an existing GGUF file (less common)
                elif (
                    self.model_id
                    and self.model_id.lower().endswith(".gguf")
                    and os.path.exists(self.model_id)
                ):
                    model_file_to_load = self.model_id
                    logger.info(
                        f"Using model file directly from MODEL_ID path: {model_file_to_load}"
                    )
                # Check 3: Resolve MODEL_ID using AVAILABLE_MODELS dictionary
                elif self.model_id in AVAILABLE_MODELS:
                    model_filename = AVAILABLE_MODELS[self.model_id].get("filename")
                    if model_filename:
                        potential_path = os.path.join(MODELS_DIR, model_filename)
                        if os.path.exists(potential_path):
                            model_file_to_load = potential_path
                            logger.info(
                                f"Resolved MODEL_ID '{self.model_id}' to path: {model_file_to_load}"
                            )
                        else:
                            logger.warning(
                                f"Resolved MODEL_ID '{self.model_id}' to {potential_path}, but file not found."
                            )
                    else:
                        logger.warning(
                            f"MODEL_ID '{self.model_id}' found in config, but no filename specified."
                        )

                # Check if we found a valid path
                if not model_file_to_load:
                    # Raise error only if no valid path could be determined at all
                    raise ValueError(
                        f"Could not determine a valid model file path for llama.cpp from MODEL_PATH='{self.model_path}' or MODEL_ID='{self.model_id}'"
                    )

                logger.info(f"Loading llama.cpp model from: {model_file_to_load}")
                self.llm = Llama(
                    model_path=model_file_to_load,
                    n_ctx=self.max_model_len,
                    n_gpu_layers=-1,
                    chat_format=self.chat_format,
                    verbose=settings.LLAMA_CPP_VERBOSE,
                    n_threads=self.n_threads,
                    special_eog_token_ids=[1, 107],
                    flash_attn=settings.FLASH_ATTENTION,
                )
                # Attempt to load a tokenizer if llama.cpp model has one, or use a fallback
                if hasattr(self.llm, "tokenizer") and self.llm.tokenizer is not None:
                    self.tokenizer = self.llm

                # Determine the identifier for the loaded model for logging purposes
                # Prefer the actual path used, then the model_id if it matches, then a generic identifier.
                loaded_model_identifier = model_file_to_load
                if self.model_id and self.model_id in model_file_to_load:
                    loaded_model_identifier = self.model_id
                elif (
                    self.model_id
                ):  # If model_id is set but doesn't match path, log both for clarity
                    loaded_model_identifier = (
                        f"{model_file_to_load} (config MODEL_ID: {self.model_id})"
                    )

                logger.info(
                    f"Model '{loaded_model_identifier}' loaded successfully with llama.cpp backend (n_ctx={self.llm.n_ctx()})"
                )

            else:
                logger.error(f"Unsupported LLM backend specified: {self.backend}")
                raise ValueError(f"Unsupported LLM backend: {self.backend}")

        except Exception as e:
            logger.error(
                f"Fatal error loading model with {self.backend} backend: {str(e)}",
                exc_info=True,
            )
            # Set service to a non-functional state?
            self.llm = None
            self.tokenizer = None
            self.client = None
            # Re-raise the exception to prevent the application from starting incorrectly
            raise RuntimeError(
                f"Failed to initialize LLMService backend {self.backend}"
            ) from e

    def reload_model(self, model_path: Optional[str] = None):
        """Reloads the LLM. Can load a specific model path or use latest .env config."""
        if model_path:
            logger.info(f"Reloading LLM service to specific model: {model_path}")
            # If specific path is given, update internal state directly
            self.model_path = model_path
            # We might need to infer model_id if other parts of the service rely on it
            # For now, assume model_path is sufficient for _load_model
            self.model_id = None  # Or try to infer from AVAILABLE_MODELS based on path?
        else:
            logger.info(
                "Reloading LLM service based on current environment configuration..."
            )
            # Update config from .env if no specific path is provided
            self._update_config_from_env()

        # Load the model using the now updated self.model_path etc.
        self._load_model()
        logger.info("Model reload process completed.")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        try:
            if self.backend == "llama.cpp":
                # For llama.cpp, use token_count from the model
                return len(self.llm.tokenize(text.encode("utf-8")))
            elif self.backend == "azure":
                if self.tokenizer:
                    return len(self.tokenizer.encode(text))
                else:
                    # Use tiktoken for Azure if tokenizer is not available
                    encoding = tiktoken.get_encoding(
                        "cl100k_base"
                    )  # Common for Azure models
                    return len(encoding.encode(text))
            elif self.backend in ["vllm", "ollama"]:  # Added ollama here
                # For vLLM and Ollama, use the tokenizer if available
                if self.tokenizer:
                    return len(self.tokenizer.encode(text))
                else:
                    # Fallback to tiktoken if tokenizer is not available for vLLM/Ollama
                    # This might not be perfect but better than char count
                    logger.warning(
                        "Tokenizer not found for vLLM/Ollama, using tiktoken for token count."
                    )
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
            else:  # General case if backend unknown or tokenizer not set for it
                logger.warning(
                    f"Unknown backend {self.backend} or tokenizer not set, using tiktoken for token count."
                )
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
        except AttributeError as e:
            logger.warning(
                f"AttributeError in count_tokens (e.g. tokenizer not ready): {e}. Falling back to tiktoken."
            )
            # Fallback token counting if tokenizer is not available
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(
                f"Unexpected error in count_tokens: {e}. Falling back to tiktoken."
            )
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    def generate_sync(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        top_k: int = 40,
    ) -> str:
        """Synchronous generation method - actual implementation."""
        logger.debug(f"generate_sync called for backend {self.backend}")
        if self.backend == "llama.cpp":
            if not self.llm:
                raise RuntimeError("Llama.cpp model not loaded.")

            if messages:
                # Use chat completion if messages are provided
                completion = self.llm.create_chat_completion(
                    messages=self._massage_messages_for_template(messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop or [],
                    top_k=top_k,
                )
                return completion["choices"][0]["message"]["content"]
            elif prompt:
                # Use standard completion if only prompt is provided
                completion = self.llm(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop or [],
                    top_k=top_k,
                )
                return completion["choices"][0]["text"]
            else:
                raise ValueError("Either prompt or messages must be provided.")

        # Add similar blocks for vllm, ollama, azure backends based on
        # the logic previously in the duplicate generate methods.
        elif self.backend == "ollama":
            # ... Ollama logic using requests ...
            # Example (needs refinement based on original code):
            if messages:
                payload = {
                    "model": self.model_id,
                    "messages": self._massage_messages_for_template(messages),
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                        "stop": stop,
                        "top_k": top_k,
                    },
                }
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat", json=payload
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
            elif prompt:
                payload = {
                    "model": self.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                        "stop": stop,
                        "top_k": top_k,
                    },
                }
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate", json=payload
                )
                response.raise_for_status()
                return response.json()[
                    "response"
                ]  # Adjust based on actual API response
            else:
                raise ValueError("Either prompt or messages must be provided.")

        elif self.backend == "azure":
            # ... Azure logic using self.client ...
            if messages:
                response = self.client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=self._massage_messages_for_template(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                return response.choices[0].message.content
            elif prompt:
                # Azure might not have a direct text completion endpoint like this
                # Typically use chat completion even for single prompts
                response = self.client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                return response.choices[0].message.content
            else:
                raise ValueError("Either prompt or messages must be provided.")

        # Add vLLM logic if it was present
        # elif self.backend == "vllm":
        #    ...

        else:
            raise NotImplementedError(
                f"Backend '{self.backend}' synchronous generation not implemented."
            )

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        top_k: int = 40,
    ) -> str:
        """Asynchronous generation method wrapper using run_in_executor with timeout."""
        # (Checklist Item 3-C, G1, G2)
        loop = asyncio.get_running_loop()
        # Use a lambda to pass arguments correctly to the sync method
        func_call = lambda: self.generate_sync(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            top_k=top_k,
        )
        try:
            # Run in the dedicated LLM executor with timeout from settings
            result = await asyncio.wait_for(
                loop.run_in_executor(self.llm_executor, func_call),
                timeout=self.settings.LLM_TIMEOUT,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                f"LLM generation timed out after {self.settings.LLM_TIMEOUT} seconds."
            )
            # Re-raise TimeoutError so it can be caught upstream (e.g., in RagEngine or API route)
            raise
        except Exception as e:
            logger.error(
                f"Error during run_in_executor for LLM generation: {e}", exc_info=True
            )
            raise  # Re-raise other exceptions

    async def generate_with_json_output(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Generate response enforcing JSON output matching schema (Llama.cpp backend only)."""
        # Add instructions to the prompt
        json_prompt = f"""
        {prompt}

        You must respond with a JSON object matching this schema:
        {json.dumps(json_schema, indent=2)}

        Response (JSON only):
        """

        # Set stop sequence for JSON generation
        stop_sequences = ["\n\n"]

        # Generate JSON text
        json_text = await self.generate(
            json_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )

        # Parse JSON output
        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError:
            # If parsing fails, try to fix common JSON errors
            try:
                # Try to extract just the JSON part
                json_text = json_text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text.replace("```json", "", 1)
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                json_text = json_text.strip()

                result = json.loads(json_text)
                return result
            except:
                raise ValueError(f"Failed to parse JSON output: {json_text}")

    def _massage_messages_for_template(self, messages: list[dict]) -> list[dict]:
        """Convert messages to the format expected by the backend's chat template."""
        massaged = []
        system_prompt_found = False
        downgrade_system = False

        # Check if system role needs downgrade (Checklist Item 3-D)
        if "gemma" in (self.model_id or "").lower():
            downgrade_system = True

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not role or not content:
                continue

            if role == "system":
                if (
                    system_prompt_found
                ):  # Only one system prompt allowed, typically at the start
                    logger.warning(
                        "Multiple system prompts found, skipping subsequent ones."
                    )
                    continue
                system_prompt_found = True
                if downgrade_system:
                    logger.debug(
                        "Downgrading system message to user message for Gemma model."
                    )
                    # Prepend indication or merge with first user message if possible
                    # Simple approach: make it a user message
                    massaged.append(
                        {"role": "user", "content": f"System Instruction: {content}"}
                    )
                else:
                    massaged.append({"role": "system", "content": content})
            elif role in ["user", "assistant"]:
                massaged.append({"role": role, "content": content})
            else:
                logger.warning(f"Unknown role '{role}' in message, skipping.")
        return massaged

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,  # Note: Default changed here in retry version vs others
        stop: Optional[List[str]] = None,
        # Added top_p and top_k to match other signatures
        top_p: float = 0.9,
        top_k: int = 40,
        json_output: bool = False,  # Kept json_output specific to this retry method?
    ) -> Union[str, Dict]:
        """Main asynchronous generation method with retry logic."""
        # This method should now likely call generate_async which uses run_in_executor
        # Or directly call generate_sync within run_in_executor if generate_async is removed/refactored.
        # Let's make it call generate_async:

        if json_output:
            # Handle JSON output generation (if applicable, might only work for specific backends/methods)
            # For now, assuming it calls a specific method or is part of generate_sync
            # This part needs clarification based on how generate_with_json_output was used.
            # Placeholder:
            logger.warning(
                "JSON output requested in retry generate, ensure backend supports it."
            )
            # Fall through to normal generation for now, assuming generate_sync handles it?
            # Or call generate_with_json_output here if that's the intent?
            # return await self.generate_with_json_output(...) # Requires prompt and schema

        # Call the primary async (non-blocking) method
        try:
            return await self.generate_async(
                prompt=prompt,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(
                f"Error during LLM generation (attempt before retry): {e}",
                exc_info=True,
            )
            raise  # Re-raise to trigger tenacity retry

    async def check_context_grounding(
        self,
        question: str,
        context: str,
        answer: str,  # Added answer param based on usage
    ) -> str:
        """
        Uses LLM to check if the answer is grounded in the provided context.

        Returns:
            "grounded", "not_grounded", or "error"
        """
        grounding_prompt = f"""Context: {context}
Question: {question}
Answer: {answer}

Is the Answer supported by the Context? Answer only YES or NO."""
        try:
            # Use the primary generate method (Checklist Item 3-B)
            response_text = await self.generate(
                prompt=grounding_prompt,  # Corrected kwarg
                temperature=0.01,  # Corrected kwarg
                max_tokens=10,  # Corrected kwarg
                stop=["\n"],  # Corrected kwarg
            )
            response_clean = response_text.strip().upper()
            if "YES" in response_clean:
                return "grounded"
            elif "NO" in response_clean:
                return "not_grounded"
            else:
                logger.warning(
                    f"Grounding check returned unexpected response: {response_text}"
                )
                return "error"  # Indicate unclear response
        except Exception as e:
            logger.error(f"Error during grounding check: {e}")
            return "error"

    def free_resources(self):
        """Placeholder for freeing GPU memory or other resources if needed."""
        logger.info("Freeing LLMService resources...")
        if self.llm:
            # Specific cleanup depends heavily on the backend library
            # For llama-cpp, setting to None might be sufficient if refs are managed
            # For vLLM or others, specific unload methods might exist
            del self.llm
            self.llm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.client:
            # OpenAI client doesn't usually need explicit closing
            self.client = None

        # Shutdown the dedicated executor (G1 / G2 / P2)
        if self.llm_executor:
            logger.info("Shutting down LLM ThreadPoolExecutor...")
            self.llm_executor.shutdown(wait=False)  # Don't block shutdown
            self.llm_executor = None
            logger.info("LLM ThreadPoolExecutor shut down.")

        # Consider torch.cuda.empty_cache() if using PyTorch directly
        logger.info("LLMService resources released (best effort).")

    def get_token_count(self, text: str) -> int:
        """Estimate token count using the loaded tokenizer."""
        # This duplicates count_tokens, consolidating.
        return self.count_tokens(text)

    async def aclose(self):
        """Async cleanup method for shutdown event."""
        self.free_resources()


# Example instantiation (usually done via dependency injection)
# llm_service = LLMService()
