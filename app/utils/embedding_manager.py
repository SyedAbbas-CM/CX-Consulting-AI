"""
Embedding Model Manager

A flexible utility for managing different embedding models with graceful fallbacks.
"""

import hashlib
import importlib.util
import logging
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch

# Configure logger
logger = logging.getLogger("cx_consulting_ai.embedding_manager")


class EmbeddingModelType(Enum):
    """Enum for supported embedding model types."""

    BGE = "bge"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    ONNX = "onnx"
    LOCAL = "local"


class EmbeddingManager:
    """Manager for embedding models with fallback mechanisms."""

    def __init__(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_offline_fallback: bool = True,
    ):
        """
        Initialize the embedding manager.

        Args:
            model_type: Type of embedding model (bge, sentence_transformers, onnx, local)
            model_name: Name or path of the model
            device: Device to run the model on (cpu, cuda, mps)
            use_offline_fallback: Whether to use local fallback when online models fail
        """
        # Set default values from environment or parameters
        self.model_type = model_type or os.getenv("EMBEDDING_TYPE", "bge").lower()

        # Set model name based on type if not provided
        if not model_name:
            if self.model_type == EmbeddingModelType.BGE.value:
                self.model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            else:
                self.model_name = os.getenv(
                    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                )
        else:
            self.model_name = model_name

        self.use_offline_fallback = use_offline_fallback

        # Determine device: Priority -> explicit arg > ENV VAR > auto-detect
        requested_device = device or os.getenv("EMBEDDING_DEVICE")
        if requested_device:
            self.device = requested_device.lower()
            logger.info(f"Using device specified via constructor or ENV: {self.device}")
        else:
            self.device = self._detect_device()
            logger.info(f"Using auto-detected device: {self.device}")

        # If device is MPS, log a specific warning/info about potential issues and CPU override
        if self.device == "mps":
            logger.warning(
                "Device selected is MPS (Apple Silicon). If embeddings hang, set EMBEDDING_DEVICE=cpu in your environment and restart."
            )
            # Optional: Force CPU if MPS is known to be problematic for the specific model?
            # E.g.: if self.model_name == "BAAI/bge-large-en-v1.5": self.device = "cpu"

        # Initialize model
        self.model = None
        self.embed_documents = None
        self.embed_query = None
        self.actual_dimension: Optional[int] = None

        # Try to initialize the model
        try:
            if self.model_type == EmbeddingModelType.BGE.value:
                self._init_bge_model()
            elif self.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS.value:
                self._init_sentence_transformers_model()
            elif self.model_type == EmbeddingModelType.ONNX.value:
                self._init_onnx_model()
            else:
                logger.warning(
                    f"Unknown model type: {self.model_type}, using local fallback"
                )
                self._init_local_fallback()
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            if self.use_offline_fallback:
                logger.warning("Falling back to local embedding function")
                self._init_local_fallback()
            else:
                raise

        # Ensure dimension is set after initialization attempt
        if self.actual_dimension is None and not self.use_offline_fallback:
            raise RuntimeError(
                "Failed to determine embedding dimension after model initialization."
            )
        elif self.actual_dimension is None and self.use_offline_fallback:
            logger.warning("Using fallback dimension size for local fallback.")
            self.actual_dimension = 384  # Set fallback dimension if needed

            # --- ADD WARM-UP ---
            if (
                self.model
                and callable(getattr(self.model, "encode", None))
                and self.device == "mps"
            ):
                try:
                    logger.info(
                        f"Performing MPS warm-up encode on device {self.device}..."
                    )
                    _ = self.model.encode("warm-up", normalize_embeddings=True)
                    logger.info("MPS warm-up encode completed.")
                except Exception as wu_err:
                    logger.warning(f"MPS warm-up failed: {wu_err}")
            # --- END WARM-UP ---

    def _detect_device(self) -> str:
        """Detect the best available device."""
        try:
            if torch.backends.mps.is_available():
                logger.info("MPS is available, using Apple Silicon accelerator")
                return "mps"
            elif torch.cuda.is_available():
                logger.info("CUDA is available, using GPU")
                return "cuda"
            else:
                logger.info("Using CPU for embeddings")
                return "cpu"
        except:
            logger.info("Could not detect PyTorch devices, using CPU")
            return "cpu"

    def _init_bge_model(self):
        """Initialize BGE model (specialized for RAG)."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading BGE embedding model: {self.model_name} onto device: {self.device}"
            )
            self.model = SentenceTransformer(self.model_name, device=self.device)

            if self.device == "mps":
                logger.info("Casting BGE model to torch.float16 for MPS device.")
                self.model = self.model.to(torch.float16)

            # Get actual dimension
            try:
                self.actual_dimension = self.model.get_sentence_embedding_dimension()
                if self.actual_dimension:
                    logger.info(
                        f"Detected embedding dimension: {self.actual_dimension}"
                    )
                else:
                    raise ValueError("Model did not return a valid dimension.")
            except Exception as dim_err:
                logger.error(
                    f"Could not automatically determine embedding dimension: {dim_err}. Check model compatibility."
                )
                raise

            # BGE performs better with query instruction for retrieval
            query_instruction = (
                "Represent this sentence for searching relevant passages: "
            )

            # Define embedding functions
            self.embed_documents = lambda texts: self.model.encode(
                texts, show_progress_bar=True, normalize_embeddings=True
            ).tolist()

            # Add instruction to query and ensure output is List[List[float]]
            self.embed_query = lambda text: [
                self.model.encode(
                    query_instruction + text, normalize_embeddings=True
                ).tolist()
            ]

            logger.info(f"BGE embedding initialized on {self.device}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Please install it: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BGE model: {str(e)}")
            raise

    def _init_sentence_transformers_model(self):
        """Initialize SentenceTransformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading SentenceTransformers model: {self.model_name} onto device: {self.device}"
            )
            self.model = SentenceTransformer(self.model_name, device=self.device)

            if self.device == "mps":
                logger.info(
                    "Casting SentenceTransformers model to torch.float16 for MPS device."
                )
                self.model = self.model.to(torch.float16)

            # Get actual dimension
            try:
                self.actual_dimension = self.model.get_sentence_embedding_dimension()
                if self.actual_dimension:
                    logger.info(
                        f"Detected embedding dimension: {self.actual_dimension}"
                    )
                else:
                    raise ValueError("Model did not return a valid dimension.")
            except Exception as dim_err:
                logger.error(
                    f"Could not automatically determine embedding dimension: {dim_err}. Check model compatibility."
                )
                raise

            # Define embedding functions
            self.embed_documents = lambda texts: self.model.encode(
                texts, show_progress_bar=True, normalize_embeddings=True
            ).tolist()

            # Ensure output is List[List[float]]
            self.embed_query = lambda text: [
                self.model.encode(text, normalize_embeddings=True).tolist()
            ]

            logger.info(f"SentenceTransformers embedding initialized on {self.device}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Please install it: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformers model: {str(e)}")
            raise

    def _init_onnx_model(self):
        """Initialize ONNX model for faster inference."""
        try:
            if not importlib.util.find_spec("onnxruntime"):
                raise ImportError("onnxruntime is not installed")

            import numpy as np
            import onnxruntime
            from transformers import AutoTokenizer

            # Check for model file
            if not os.path.exists(self.model_name):
                raise ValueError(f"ONNX model file not found: {self.model_name}")

            # Get tokenizer name from environment or use default
            default_onnx_tokenizer = (
                "BAAI/bge-small-en-v1.5"  # Default if nothing else is found
            )
            tokenizer_name = os.getenv("ONNX_TOKENIZER", default_onnx_tokenizer)

            # Attempt to infer tokenizer from model_name (ONNX path) if tokenizer_name is still the default
            # and ONNX_TOKENIZER was not explicitly set to something else.
            # This helps if ONNX_TOKENIZER is missing but the ONNX model path has a clear structure.
            if (
                tokenizer_name == default_onnx_tokenizer
                and "ONNX_TOKENIZER" not in os.environ
            ):
                try:
                    # Example path: models/onnx/BAAI/bge-large-en-v1.5/model.onnx
                    parts = self.model_name.split(os.sep)
                    if "onnx" in parts and parts.index("onnx") + 2 < len(parts):
                        # Assumes structure like /onnx/AUTHOR/MODEL_NAME_SUFFIX/
                        author = parts[parts.index("onnx") + 1]
                        model_suffix = parts[parts.index("onnx") + 2]
                        inferred_tokenizer = f"{author}/{model_suffix}"
                        # Basic check if it looks like a valid model name
                        if (
                            "/" in inferred_tokenizer
                            and not inferred_tokenizer.endswith("/")
                        ):
                            logger.info(
                                f"Attempting to infer ONNX tokenizer from path: {inferred_tokenizer}"
                            )
                            # Try to load tokenizer to validate
                            AutoTokenizer.from_pretrained(inferred_tokenizer)
                            tokenizer_name = inferred_tokenizer
                        else:
                            logger.info(
                                "Could not confidently infer tokenizer from ONNX model path, using default or ONNX_TOKENIZER env var."
                            )
                    else:
                        logger.info(
                            "ONNX model path does not match expected structure for tokenizer inference."
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not auto-infer tokenizer from ONNX path ({self.model_name}): {e}. Using default or ONNX_TOKENIZER env var: {tokenizer_name}"
                    )

            if (
                tokenizer_name == default_onnx_tokenizer
                and "bge-large" in self.model_name.lower()
            ):
                logger.warning(
                    f"ONNX model path '{self.model_name}' seems to be for a large model, but tokenizer is '{tokenizer_name}'. Consider setting ONNX_TOKENIZER=BAAI/bge-large-en-v1.5 in your .env file."
                )

            logger.info(f"Loading ONNX model: {self.model_name}")
            logger.info(f"Using tokenizer: {tokenizer_name}")

            # Create session
            sess_options = onnxruntime.SessionOptions()
            self.model = onnxruntime.InferenceSession(
                self.model_name,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Get actual dimension
            try:
                # Run a dummy input to get output shape
                dummy_text = "dimension check"
                encoded_input = self.tokenizer(dummy_text, return_tensors="np")
                model_inputs = {k: v for k, v in encoded_input.items()}
                outputs = self.model.run(None, model_inputs)
                output_shape = outputs[0].shape
                # Shape is usually (batch_size, sequence_length, dimension) or (batch_size, dimension) for pooled output
                if len(output_shape) == 2:  # Pooled output
                    self.actual_dimension = output_shape[1]
                elif (
                    len(output_shape) == 3
                ):  # May need pooling - assume model pools or take first token?
                    # This might need adjustment based on the specific ONNX model's output
                    logger.warning(
                        "ONNX output shape suggests sequence embeddings. Assuming pooled dimension is last."
                    )
                    self.actual_dimension = output_shape[-1]
                else:
                    raise ValueError(f"Unexpected ONNX output shape: {output_shape}")
                logger.info(
                    f"Detected ONNX embedding dimension: {self.actual_dimension}"
                )
            except Exception as dim_err:
                logger.error(
                    f"Could not automatically determine ONNX embedding dimension: {dim_err}. Check model output."
                )
                raise

            # BGE instruction
            query_instruction = (
                "Represent this sentence for searching relevant passages: "
            )

            # Define embedding functions
            def encode_texts(texts, is_query=False):
                # Apply instruction for queries (BGE specific, adapt if model is not BGE-like)
                # For now, this ONNX path uses BGE tokenizer by default, so instruction is relevant.
                # if is_query and self.model_type == EmbeddingModelType.BGE.value: # More specific check if needed
                if is_query:
                    if isinstance(texts, str):
                        texts = [query_instruction + texts]  # Becomes list of one
                    elif isinstance(texts, list):
                        texts = [query_instruction + t for t in texts]

                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="np",
                    max_length=512,  # Default max length for many sentence transformers
                )
                # Run inference
                outputs = self.model.run(None, dict(inputs))
                # Get last_hidden_state
                embeddings = outputs[0]
                # Mean pooling (CLS token often not best for sentence embeddings)
                # Take mean of all token embeddings for each sentence
                # Create attention mask for pooling (np.where(inputs['attention_mask']...))
                attention_mask = inputs["attention_mask"]
                masked_embeddings = embeddings * np.expand_dims(attention_mask, axis=-1)
                summed_embeddings = np.sum(masked_embeddings, axis=1)
                counts = np.sum(attention_mask, axis=1, keepdims=True)
                pooled_embeddings = summed_embeddings / np.maximum(
                    counts, 1e-9
                )  # Avoid division by zero

                # Normalize embeddings
                norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
                normalized_embeddings = pooled_embeddings / np.maximum(norms, 1e-9)

                result_list = normalized_embeddings.tolist()
                # If it was originally a single query string, ensure it's List[List[float]]
                if (
                    is_query
                    and isinstance(texts, list)
                    and len(texts) == 1
                    and isinstance(texts[0], str)
                    and texts[0].startswith(query_instruction)
                ):
                    return [
                        result_list[0]
                    ]  # It's already a list of one embedding, which is List[float]. So wrap again.
                return result_list

            self.embed_documents = lambda texts_list: encode_texts(
                texts_list, is_query=False
            )
            self.embed_query = lambda text_str: encode_texts(
                text_str, is_query=True
            )  # encode_texts handles single string to list and returns List[List[float]] if query

            logger.info("ONNX embedding initialized")
        except ImportError as e:
            logger.error(f"Required packages not installed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ONNX model: {str(e)}")
            raise

    def _init_local_fallback(self):
        """Initialize with a simple local embedding function as fallback."""
        import hashlib

        logger.info("Using simple local embedding function (offline mode)")
        self.model_type = EmbeddingModelType.LOCAL.value
        self.model_name = "local_fallback_hasher"

        # Define a simple embedding function using hash values
        def simple_embedding(text: str) -> List[float]:
            # O3 Fix 7.1: Ensure input is string
            if not isinstance(text, str):
                logger.warning(
                    f"Local fallback received non-string input: {type(text)}"
                )
                text = str(text)  # Attempt conversion

            # Create a hash of the text
            hash_object = hashlib.md5(text.encode())
            hash_hex = hash_object.hexdigest()

            # Convert hash to a vector of floats
            vector = []
            for i in range(0, len(hash_hex), 2):
                if i + 2 <= len(hash_hex):
                    hex_pair = hash_hex[i : i + 2]
                    value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
                    vector.append(value)

            # Pad to 384 dimensions for a reasonably sized vector
            # O3 Fix: Explicitly set fallback dimension
            fallback_dim = 384
            vector = vector + [0.0] * (fallback_dim - len(vector))
            # Ensure dimension is set even in fallback
            self.actual_dimension = fallback_dim
            return vector

        # O3 Fix 7.1: Implement proper loop for embed_documents
        def embed_docs(texts: List[str]) -> List[List[float]]:
            return [simple_embedding(t) for t in texts]

        # O3 Fix 7.3: Ensure embed_query returns List[List[float]]
        def embed_q(text: str) -> List[List[float]]:
            return [simple_embedding(text)]

        # Set embedding functions
        self.embed_documents = embed_docs
        self.embed_query = embed_q

        logger.info(
            f"Local embedding function initialized with dimension {self.actual_dimension}"
        )

    def embed(
        self, texts: Union[str, List[str]], is_query: bool = False
    ) -> List[List[float]]:
        """Embed a single text or a list of texts."""
        if not self.embed_documents or not self.embed_query:
            logger.error("Embedding functions not initialized.")
            # Fallback to basic local if primary init failed and this is called.
            if self.use_offline_fallback and not (
                hasattr(self, "_local_fallback_active") and self._local_fallback_active
            ):
                self._init_local_fallback()
            elif not self.use_offline_fallback:
                raise RuntimeError("Embedding functions not available and no fallback.")

        if is_query:
            # embed_query is designed to take a single string and return List[List[float]]
            if isinstance(texts, str):
                return self.embed_query(texts)
            elif isinstance(texts, list):
                # If a list is passed to a query, embed one by one and concatenate
                # This is less efficient but handles API for embed_query which expects single text.
                # Or, adapt embed_query to handle List[str] as well.
                # For now, let's assume embed_query is for single text query
                if len(texts) == 1:
                    return self.embed_query(texts[0])
                else:
                    # This case is ambiguous for a single self.embed_query call designed for one text.
                    # For multiple query texts, one might call self.embed_documents or loop self.embed_query.
                    # Sticking to current structure: if is_query is True, it implies a typical single query context.
                    logger.warning(
                        "embed(is_query=True) called with a list of texts. Processing first text only for embed_query."
                    )
                    return self.embed_query(texts[0])
            else:
                raise TypeError("Input to embed must be str or List[str]")
        else:
            # embed_documents is designed to take List[str] and return List[List[float]]
            # It can also handle single str by wrapping it.
            if isinstance(texts, str):
                return self.embed_documents([texts])
            elif isinstance(texts, list):
                return self.embed_documents(texts)
            else:
                raise TypeError("Input to embed must be str or List[str]")

    def get_embedding_function(self, for_queries: bool = False) -> Callable:
        """Return the appropriate embedding function."""
        if for_queries:
            if self.embed_query is None:
                logger.error("Query embedding function not initialized.")
                # Fallback or raise?
                raise RuntimeError("Query embedding function unavailable.")
            return self.embed_query
        else:
            if self.embed_documents is None:
                logger.error("Document embedding function not initialized.")
                raise RuntimeError("Document embedding function unavailable.")
            return self.embed_documents

    @property
    def dimension_size(self) -> Optional[int]:
        """Returns the actual dimension size of the loaded embedding model."""
        return self.actual_dimension


def create_embedding_manager(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> EmbeddingManager:
    """
    Factory function to create an EmbeddingManager instance.
    Allows passing additional kwargs for future flexibility.
    """
    # Determine model_type and model_name from environment if not provided
    effective_model_type = model_type or os.getenv("EMBEDDING_TYPE", "bge").lower()

    effective_model_name = model_name
    if not effective_model_name:
        if effective_model_type == EmbeddingModelType.BGE.value:
            effective_model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        else:
            effective_model_name = os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )

    logger.info(
        f"Creating EmbeddingManager with type: {effective_model_type}, name: {effective_model_name}, device: {device}"
    )
    return EmbeddingManager(
        model_type=effective_model_type,
        model_name=effective_model_name,
        device=device,
        **kwargs,
    )
