"""
Embedding Model Manager

A flexible utility for managing different embedding models with graceful fallbacks.
"""
import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
import importlib.util

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
        use_offline_fallback: bool = True
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
                self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.model_name = model_name
        
        self.use_offline_fallback = use_offline_fallback
        
        # Auto-detect device if not provided
        self.device = device
        if not self.device:
            self.device = self._detect_device()
        
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
                logger.warning(f"Unknown model type: {self.model_type}, using local fallback")
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
             raise RuntimeError("Failed to determine embedding dimension after model initialization.")
        elif self.actual_dimension is None and self.use_offline_fallback:
             logger.warning("Using fallback dimension size for local fallback.")
             self.actual_dimension = 384 # Set fallback dimension if needed
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch
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
            
            logger.info(f"Loading BGE embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Get actual dimension
            try:
                self.actual_dimension = self.model.get_sentence_embedding_dimension()
                if self.actual_dimension:
                     logger.info(f"Detected embedding dimension: {self.actual_dimension}")
                else:
                     raise ValueError("Model did not return a valid dimension.")
            except Exception as dim_err:
                 logger.error(f"Could not automatically determine embedding dimension: {dim_err}. Check model compatibility.")
                 raise
            
            # BGE performs better with query instruction for retrieval
            query_instruction = "Represent this sentence for searching relevant passages: "
            
            # Define embedding functions
            self.embed_documents = lambda texts: self.model.encode(
                texts, 
                show_progress_bar=True,
                normalize_embeddings=True
            ).tolist()
            
            # Add instruction to query
            self.embed_query = lambda text: self.model.encode(
                query_instruction + text,
                normalize_embeddings=True
            ).tolist()
            
            logger.info(f"BGE embedding initialized on {self.device}")
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BGE model: {str(e)}")
            raise
    
    def _init_sentence_transformers_model(self):
        """Initialize SentenceTransformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading SentenceTransformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Get actual dimension
            try:
                self.actual_dimension = self.model.get_sentence_embedding_dimension()
                if self.actual_dimension:
                     logger.info(f"Detected embedding dimension: {self.actual_dimension}")
                else:
                     raise ValueError("Model did not return a valid dimension.")
            except Exception as dim_err:
                 logger.error(f"Could not automatically determine embedding dimension: {dim_err}. Check model compatibility.")
                 raise
            
            # Define embedding functions
            self.embed_documents = lambda texts: self.model.encode(
                texts, 
                show_progress_bar=True,
                normalize_embeddings=True
            ).tolist()
            
            self.embed_query = lambda text: self.model.encode(
                text,
                normalize_embeddings=True
            ).tolist()
            
            logger.info(f"SentenceTransformers embedding initialized on {self.device}")
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformers model: {str(e)}")
            raise
    
    def _init_onnx_model(self):
        """Initialize ONNX model for faster inference."""
        try:
            if not importlib.util.find_spec("onnxruntime"):
                raise ImportError("onnxruntime is not installed")
            
            import onnxruntime
            import numpy as np
            from transformers import AutoTokenizer
            
            # Check for model file
            if not os.path.exists(self.model_name):
                raise ValueError(f"ONNX model file not found: {self.model_name}")
            
            # Get tokenizer name from environment or use default
            tokenizer_name = os.getenv("ONNX_TOKENIZER", "BAAI/bge-small-en-v1.5")
            
            logger.info(f"Loading ONNX model: {self.model_name}")
            logger.info(f"Using tokenizer: {tokenizer_name}")
            
            # Create session
            sess_options = onnxruntime.SessionOptions()
            self.model = onnxruntime.InferenceSession(
                self.model_name,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Get actual dimension
            try:
                # Run a dummy input to get output shape
                dummy_text = "dimension check"
                encoded_input = self.tokenizer(dummy_text, return_tensors='np')
                model_inputs = {k: v for k, v in encoded_input.items()}
                outputs = self.model.run(None, model_inputs)
                output_shape = outputs[0].shape
                # Shape is usually (batch_size, sequence_length, dimension) or (batch_size, dimension) for pooled output
                if len(output_shape) == 2: # Pooled output
                    self.actual_dimension = output_shape[1]
                elif len(output_shape) == 3: # May need pooling - assume model pools or take first token?
                    # This might need adjustment based on the specific ONNX model's output
                    logger.warning("ONNX output shape suggests sequence embeddings. Assuming pooled dimension is last.")
                    self.actual_dimension = output_shape[-1]
                else:
                    raise ValueError(f"Unexpected ONNX output shape: {output_shape}")
                logger.info(f"Detected ONNX embedding dimension: {self.actual_dimension}")
            except Exception as dim_err:
                 logger.error(f"Could not automatically determine ONNX embedding dimension: {dim_err}. Check model output.")
                 raise
            
            # BGE instruction
            query_instruction = "Represent this sentence for searching relevant passages: "
            
            # Define embedding functions
            def encode_texts(texts, is_query=False):
                # Apply instruction for queries
                if is_query and isinstance(texts, str):
                    texts = query_instruction + texts
                
                # Tokenize
                encoded_input = self.tokenizer(
                    texts, 
                    padding=True,
                    truncation=True,
                    return_tensors='np'
                )
                
                # Run inference
                model_inputs = {k: v for k, v in encoded_input.items()}
                outputs = self.model.run(None, model_inputs)
                
                # Get embeddings (mean pooling)
                embeddings = outputs[0]
                
                # Normalize
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                return embeddings.tolist()
            
            self.embed_documents = lambda texts: encode_texts(texts)
            self.embed_query = lambda text: encode_texts(text, is_query=True)
            
            logger.info("ONNX embedding initialized")
        except ImportError as e:
            logger.error(f"Required packages not installed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ONNX model: {str(e)}")
            raise
    
    def _init_local_fallback(self):
        """Initialize simple local embedding function."""
        logger.info("Using simple local embedding function (offline mode)")
        self.actual_dimension = 384 # Set fixed dimension for fallback
        
        # Define a simple embedding function using hash values
        def simple_embedding(text):
            if isinstance(text, list):
                return [simple_embedding(t) for t in text]
            
            # Create a hash of the text
            hash_object = hashlib.md5(text.encode())
            hash_hex = hash_object.hexdigest()
            
            # Convert hash to a vector of floats
            vector = []
            for i in range(0, len(hash_hex), 2):
                if i+2 <= len(hash_hex):
                    hex_pair = hash_hex[i:i+2]
                    value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
                    vector.append(value)
            
            # Pad to desired dimension
            vector = vector + [0.0] * (self.actual_dimension - len(vector))
            return vector
        
        # Set embedding functions
        self.embed_documents = lambda texts: simple_embedding(texts)
        self.embed_query = lambda text: simple_embedding(text)
        
        logger.info(f"Local embedding function initialized with dimension {self.actual_dimension}")

    def embed(self, texts: Union[str, List[str]], is_query: bool = False) -> List[List[float]]:
        """
        Embed texts using the selected model.
        
        Args:
            texts: Text or list of texts to embed
            is_query: Whether the text is a query (vs document)
            
        Returns:
            List of embedding vectors
        """
        if is_query:
            if isinstance(texts, list):
                return [self.embed_query(t) for t in texts]
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)
    
    def get_embedding_function(self, for_queries: bool = False) -> Callable:
        """
        Get the appropriate embedding function.
        
        Args:
            for_queries: Whether to get the query embedding function
            
        Returns:
            Embedding function
        """
        if for_queries:
            return self.embed_query
        return self.embed_documents
    
    @property
    def dimension_size(self) -> Optional[int]:
        """Returns the actual dimension size of the loaded embedding model."""
        return self.actual_dimension

def create_embedding_manager(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> EmbeddingManager:
    """
    Factory function to create an embedding manager.
    
    Args:
        model_type: Type of embedding model
        model_name: Name or path of the model
        **kwargs: Additional arguments
        
    Returns:
        Configured EmbeddingManager
    """
    return EmbeddingManager(model_type=model_type, model_name=model_name, **kwargs) 