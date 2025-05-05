from typing import Dict, Any, Optional, List, Union
import os
from dotenv import load_dotenv, find_dotenv
import json
from pathlib import Path
import sys
import logging
import asyncio
from app.core.config import get_settings
import time
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed
from app.scripts.model_manager import MODELS_DIR, AVAILABLE_MODELS

# Configure logger
logger = logging.getLogger("cx_consulting_ai.llm_service")

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()

# Check which backend we're using
LLM_BACKEND = settings.LLM_BACKEND

if LLM_BACKEND == "vllm":
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
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
    
    def __init__(
        self,
        gpu_count: int = None,
        max_model_len: int = None
    ):
        """
        Initialize the LLM service configuration. 
        Reads config directly from environment/settings.
        
        Args:
            gpu_count: Initial GPU count (can be overridden by env)
            max_model_len: Initial max sequence length (can be overridden by env)
        """
        # Store initial values for GPU/length, but not model path/id
        self._initial_gpu_count = gpu_count
        self._initial_max_model_len = max_model_len
        
        self.llm = None
        self.tokenizer = None
        self.client = None # For Azure
        self.backend = None
        self.model_id = None
        self.model_path = None
        self.gpu_count = None
        self.max_model_len = None
        self.azure_endpoint = None
        self.azure_api_key = None
        self.azure_deployment = None
        self.ollama_base_url = None
        self.n_threads = settings.N_THREADS or os.cpu_count()
        self.chat_format = settings.CHAT_FORMAT

        self._update_config_from_env() # Load initial config from .env
        self._load_model() # Load the model based on initial config

    def _update_config_from_env(self):
        """Load or reload configuration from environment variables."""
        logger.info("Updating LLMService configuration from environment variables...")
        # Force reload of .env file
        env_path = find_dotenv(usecwd=True) # Find .env in CWD or parent dirs
        if not env_path:
            logger.warning(".env file not found when updating LLMService config.")
            # Try loading without path specification as a fallback
            load_dotenv(override=True) 
        else:
            logger.info(f"Loading environment variables from: {env_path}")
            load_dotenv(dotenv_path=env_path, override=True)

        # Reload settings object after loading .env
        settings = get_settings()

        self.backend = settings.LLM_BACKEND # Update backend first
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

        logger.info(f"LLMService config updated: backend={self.backend}, model_path={self.model_path}, model_id={self.model_id}, max_len={self.max_model_len}, gpu_count={self.gpu_count}")

    def _load_model(self):
        """Load the LLM based on the current configuration."""
        # Clear existing model/client first to release resources (if any)
        if self.llm:
             logger.info(f"Unloading previous {self.backend} model...")
             # Specific cleanup might be needed depending on the backend (e.g., closing connections)
             if self.backend == "llama.cpp" and hasattr(self.llm, 'close'): # Check if close method exists
                 try:
                     # llama_cpp doesn't have an explicit close/unload in newer versions?
                     # del self.llm might be enough if GC works properly
                     pass 
                 except Exception as e:
                     logger.warning(f"Could not explicitly close previous llama.cpp instance: {e}")
             self.llm = None
             self.tokenizer = None
             logger.info("Previous model unloaded.")
        if self.client: # For Azure
             self.client = None # Assuming the client doesn't need explicit closing

        logger.info(f"Loading model with backend: {self.backend}")
        logger.info(f"Using MODEL_PATH='{self.model_path}', MODEL_ID='{self.model_id}'")

        try:
            if self.backend == "vllm":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.llm = LLM(
                    model=self.model_id,
                    tensor_parallel_size=self.gpu_count,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=0.85
                )
                logger.info(f"Model '{self.model_id}' loaded successfully with vLLM backend")

            elif self.backend == "ollama":
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                except Exception:
                    logger.warning(f"Could not load tokenizer for '{self.model_id}', using gpt2 fallback.")
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                # No LLM object to load for Ollama, just check connectivity?
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags")
                    response.raise_for_status()
                    logger.info(f"Ollama backend confirmed at {self.ollama_base_url}")
                except requests.exceptions.RequestException as e:
                     logger.error(f"Failed to connect to Ollama backend at {self.ollama_base_url}: {e}")
                     # Decide if this should raise an error or just warn
                     # raise ConnectionError(f"Failed to connect to Ollama: {e}") from e

            elif self.backend == "azure":
                self.client = openai.AzureOpenAI(
                    api_key=self.azure_api_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                    azure_endpoint=self.azure_endpoint
                )
                logger.info(f"Azure OpenAI client initialized for deployment: {self.azure_deployment}")
                try:
                    # Try loading a tokenizer appropriate for the likely model type
                    # This is heuristic, might need adjustment based on common Azure models
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2") # Fallback/common tokenizer
                except Exception:
                    self.tokenizer = None
                    logger.warning("Could not load gpt2 tokenizer for Azure token counting, will use estimation.")

            elif self.backend == "llama.cpp":
                 # Import necessary items here, only when backend is llama.cpp
                 from llama_cpp import Llama

                 model_file_to_load = None
                 # Check 1: Use MODEL_PATH if it's set and points to an existing file
                 if self.model_path and os.path.exists(self.model_path):
                     model_file_to_load = self.model_path
                     logger.info(f"Using model file from MODEL_PATH: {model_file_to_load}")
                 # Check 2: Use MODEL_ID if it's a direct path to an existing GGUF file (less common)
                 elif self.model_id and self.model_id.lower().endswith('.gguf') and os.path.exists(self.model_id):
                     model_file_to_load = self.model_id
                     logger.info(f"Using model file directly from MODEL_ID path: {model_file_to_load}")
                 # Check 3: Resolve MODEL_ID using AVAILABLE_MODELS dictionary
                 elif self.model_id in AVAILABLE_MODELS:
                     model_filename = AVAILABLE_MODELS[self.model_id].get("filename")
                     if model_filename:
                          potential_path = os.path.join(MODELS_DIR, model_filename)
                          if os.path.exists(potential_path):
                              model_file_to_load = potential_path
                              logger.info(f"Resolved MODEL_ID '{self.model_id}' to path: {model_file_to_load}")
                          else:
                              logger.warning(f"Resolved MODEL_ID '{self.model_id}' to {potential_path}, but file not found.")
                     else:
                         logger.warning(f"MODEL_ID '{self.model_id}' found in config, but no filename specified.")
                 
                 # Check if we found a valid path
                 if not model_file_to_load:
                     # Raise error only if no valid path could be determined at all
                     raise ValueError(f"Could not determine a valid model file path for llama.cpp from MODEL_PATH='{self.model_path}' or MODEL_ID='{self.model_id}'")

                 logger.info(f"Loading llama.cpp model from: {model_file_to_load}")
                 self.llm = Llama(
                     model_path=model_file_to_load,
                     n_ctx=self.max_model_len,
                     n_gpu_layers=-1, # Use all available GPU layers
                     verbose=True # Enable verbose logging from llama.cpp
                 )
                 self.tokenizer = self.llm # llama.cpp object acts as its own tokenizer
                 logger.info(f"Model '{model_file_to_load}' loaded successfully with llama.cpp backend")

            else:
                 logger.error(f"Unsupported LLM backend specified: {self.backend}")
                 raise ValueError(f"Unsupported LLM backend: {self.backend}")

        except Exception as e:
            logger.error(f"Fatal error loading model with {self.backend} backend: {str(e)}", exc_info=True)
            # Set service to a non-functional state?
            self.llm = None
            self.tokenizer = None
            self.client = None
            # Re-raise the exception to prevent the application from starting incorrectly
            raise RuntimeError(f"Failed to initialize LLMService backend {self.backend}") from e

    def reload_model(self, model_path: Optional[str] = None):
        """Reloads the LLM. Can load a specific model path or use latest .env config."""
        if model_path:
            logger.info(f"Reloading LLM service to specific model: {model_path}")
            # If specific path is given, update internal state directly
            self.model_path = model_path
            # We might need to infer model_id if other parts of the service rely on it
            # For now, assume model_path is sufficient for _load_model
            self.model_id = None # Or try to infer from AVAILABLE_MODELS based on path?
        else:
            logger.info("Reloading LLM service based on current environment configuration...")
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
                    # Very rough estimation for Azure: ~4 characters per token
                    return len(text) // 4
            else:
                # For vLLM and Ollama, use the tokenizer
                return len(self.tokenizer.encode(text))
        except AttributeError:
            # Fallback token counting if tokenizer is not available
            # A very rough approximation: ~4 characters per token
            return len(text) // 4
    
    def generate_sync(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        top_k: int = 40
    ) -> str:
        """
        Generate text from the LLM (synchronous version).
        Can accept either a single prompt string or a list of messages for chat models.
        
        Args:
            prompt: The single prompt text (optional)
            messages: List of message dictionaries (optional, for chat)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stop: Optional list of stop sequences
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        if prompt and messages:
            raise ValueError("Provide either 'prompt' or 'messages', not both")
        
        try:
            if messages:
                logger.debug(f"Generating chat completion with {len(messages)} messages.")
            else:
                logger.debug(f"Generating completion with prompt length: {len(prompt)}")
            
            if self.backend == "vllm":
                if not prompt:
                    raise NotImplementedError("vLLM chat messages not implemented yet")
                sampling_params = SamplingParams(
                    temperature=temperature, 
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop or []
                )
                outputs = self.llm.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                return generated_text or ""
            
            elif self.backend == "ollama":
                if not prompt:
                    raise NotImplementedError("Ollama chat messages not implemented yet")
                payload = {
                    "model": os.path.basename(self.model_id).split('.')[0] if '.gguf' in self.model_id else self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "stop": stop or []
                }
                response = requests.post(f"{self.ollama_base_url}/api/generate", json=payload)
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    raise Exception(f"Error generating text: {response.text}")
            
            elif self.backend == "azure":
                if not prompt:
                    raise NotImplementedError("Azure chat messages not implemented yet")
                try:
                    response = self.client.completions.create(
                        model=self.azure_deployment,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop,
                        n=1
                    )
                    if response.choices and len(response.choices) > 0:
                        return response.choices[0].text.strip()
                    else:
                        logger.warning("No valid output from Azure OpenAI")
                        return ""
                except Exception as e:
                    logger.error(f"Error generating text with Azure OpenAI: {str(e)}")
                    return f"I apologize, but I encountered an error: {str(e)}"
            
            elif self.backend == "llama.cpp":
                try:
                    if messages:
                        logger.debug(f"Using create_chat_completion with {len(messages)} messages.")
                        # Use a slightly higher temperature for llama.cpp chat for better responsiveness on simple inputs
                        llama_chat_temp = max(temperature, 0.2) # Ensure temp is at least 0.2
                        
                        # Add Gemma specific stop tokens
                        llama_stop = stop or []
                        if "<end_of_turn>" not in llama_stop:
                            llama_stop.append("<end_of_turn>")
                        # Optionally add start token too, might help sometimes
                        # if "<start_of_turn>" not in llama_stop:
                        #     llama_stop.append("<start_of_turn>")

                        logger.debug(f"Parameters: max_tokens={max_tokens}, temp={llama_chat_temp}, top_p={top_p}, top_k={top_k}, stop={llama_stop}")
                        start_time = time.time()
                        try:
                            # --- Massage messages just before the call --- 
                            if isinstance(messages, list):
                                messages = self._massage_messages_for_template(messages)
                            # ---------------------------------------------
                            output = self.llm.create_chat_completion(
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=llama_chat_temp,
                                top_p=top_p,
                                top_k=top_k,
                                stop=llama_stop,
                            )
                            end_time = time.time()
                            logger.debug(f"llama.cpp create_chat_completion took {end_time - start_time:.4f} seconds.")
                            logger.debug(f"Raw output from create_chat_completion: {output}")
                        except Exception as llama_error:
                            logger.error(f"Error during llama_cpp.create_chat_completion: {llama_error}", exc_info=True)
                            # Re-raise or return a specific error message
                            raise llama_error # Or return a user-friendly error message

                        if output and output.get("choices") and len(output["choices"]) > 0:
                            # Check if message and content exist before accessing
                            message_data = output["choices"][0].get("message", {})
                            response_text = message_data.get("content", "") # Use .get for safety

                            # response_text = output["choices"][0]["message"]["content"] # Original line
                            logger.debug(f"Generated chat response length: {len(response_text)}")
                            return response_text if response_text is not None else "" # Ensure return string
                        else:
                            logger.warning(f"No valid choices in llama.cpp chat completion output: {output}")
                            return "I apologize, but I couldn't generate a response. Please try again."
                    elif prompt:
                        logger.debug(f"Using standard call with prompt length: {len(prompt)}")
                        try:
                             output = self.llm(
                                 prompt,
                                 max_tokens=max_tokens,
                                 temperature=temperature,
                                 top_p=top_p,
                                 top_k=top_k,
                                 stop=stop or [],
                                 echo=False
                             )
                        except Exception as llama_error:
                            logger.error(f"Error during llama_cpp standard call: {llama_error}", exc_info=True)
                            # Re-raise or return a specific error message
                            raise llama_error # Or return a user-friendly error message

                        if isinstance(output, str):
                            return output
                        elif isinstance(output, dict) and "choices" in output and len(output["choices"]) > 0:
                            response_text = output["choices"][0]["text"]
                            logger.debug(f"Generated completion response length: {len(response_text)}")
                            return response_text
                        else:
                            logger.warning(f"No valid output from LLM standard call: {output}")
                            return "I apologize, but I couldn't generate a response. Please try again."
                except Exception as e:
                    # Catch errors from param setup or re-raised llama errors
                    logger.error(f"Exception during llama.cpp generation setup or execution: {str(e)}", exc_info=True)
                    return "I apologize, but I encountered an error while generating a response."
            
            return "I apologize, but I couldn't generate a response at this time."
        
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating a response."
    
    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        top_k: int = 40
    ) -> str:
        """
        Generate text from the LLM (asynchronous version).
        
        Args:
            prompt: The prompt text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stop: Optional list of stop sequences
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if self.backend == "azure":
            # Use an async approach for Azure OpenAI
            try:
                # Run in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.generate_sync(
                        prompt=prompt,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        top_k=top_k
                    )
                )
            except Exception as e:
                logger.error(f"Error in async Azure OpenAI generation: {str(e)}")
                return f"I apologize, but I encountered an error: {str(e)}"
        else:
            # For now, other backends just use the synchronous version
            # In the future, this could be optimized for true async behavior
            return self.generate_sync(
                prompt=prompt,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                top_k=top_k
            )
    
    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        top_k: int = 40
    ) -> str:
        """
        Generate text from the LLM (asynchronous interface).
        Delegates to generate_async, accepting either prompt or messages.
        """
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        if prompt and messages:
            raise ValueError("Provide either 'prompt' or 'messages', not both")

        return await self.generate_async(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            top_k=top_k
        )
    
    async def generate_with_json_output(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON output based on the schema.
        
        Args:
            prompt: The prompt text
            json_schema: The JSON schema to follow
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            JSON object conforming to the schema
        """
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
            stop=stop_sequences
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

    # ------------------------------------------------------------------
    # Some GGUF chat templates (eg. Gemma-IT) forbid "system" messages.
    # We silently map them to "user" so callers don't have to care.
    # ------------------------------------------------------------------
    def _massage_messages_for_template(self, messages: list[dict]) -> list[dict]:
        """Converts system messages to user messages if the model template forbids system roles."""
        # Check if self.llm and its chat_format exist before accessing template
        # Safely check if llm exists and has chat_format with a template attribute
        try:
            template_str = getattr(getattr(self.llm, 'chat_format', None), 'template', None)
            if not template_str or "System role not supported" not in template_str:
                return messages  # Nothing to do
        except Exception:
             # If any attribute access fails, assume no massage needed
            return messages

        logger.debug(f"Model template for {self.model_id} forbids system role. Massaging messages.")
        fixed: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                logger.debug(f"Downgrading system message to user for model {self.model_id}")
                fixed.append({"role": "user", "content": m["content"]})
            else:
                fixed.append(m)
        return fixed

    def load_model(self):
        """Loads or reloads the LLM model based on current settings."""
        self._load_model()
        logger.info("Model loaded successfully.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def generate(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, temperature: float = 0.1, max_tokens: int = 1024, stop: Optional[List[str]] = None, json_output: bool = False) -> Union[str, Dict]:
        """Generates text using the loaded LLM (async wrapper)."""
        if not self.llm:
            logger.error("LLM not loaded. Cannot generate text.")
            raise ValueError("LLM model is not loaded.")

        if not prompt and not messages:
             raise ValueError("Either prompt or messages must be provided.")
        if prompt and messages:
            raise ValueError("Provide either prompt or messages, not both.")

        if prompt:
            # Convert single prompt to message format if needed by backend/model
            # For llama-cpp, sending messages is generally preferred
            messages = [{"role": "user", "content": prompt}]

        logger.info(f"Generating completion. Temp: {temperature}, Max Tokens: {max_tokens}") 
        start_time = time.time()
        
        generation_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens > 0 else None, # Handle max_tokens=0 or less
            "stop": stop or [],
        }
        if json_output:
            generation_kwargs["response_format"] = {"type": "json_object"}
            logger.info("Requesting JSON output format.")
            
        try:
            # --- Massage messages just before the call --- 
            if isinstance(messages, list):
                messages = self._massage_messages_for_template(messages)
            # ---------------------------------------------
            output = self.llm.create_chat_completion(
                messages=messages, 
                **generation_kwargs
            )
            # ... rest of the method ... 

            # Use a ThreadPoolExecutor to run the sync method in a separate thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.generate_sync, prompt, messages, temperature, max_tokens, stop, json_output)
                result = future.result() # Wait for the sync method to complete
            return result
        except Exception as e:
            logger.error(f"Error during async generation execution: {e}", exc_info=True)
            raise 