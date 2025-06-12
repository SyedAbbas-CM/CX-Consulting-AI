# app/core/config.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(validate_assignment=True)

    # App settings
    APP_NAME: str = "CX Consulting AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # Production optimization settings
    PRODUCTION_MODE: bool = Field(default=False, env="PRODUCTION_MODE")
    DISABLE_REQUEST_LOGGING: bool = Field(default=False, env="DISABLE_REQUEST_LOGGING")
    DISABLE_AUTH_LOGGING: bool = Field(default=False, env="DISABLE_AUTH_LOGGING")
    DISABLE_FILE_LOGGING: bool = Field(default=False, env="DISABLE_FILE_LOGGING")

    # Deployment mode
    DEPLOYMENT_MODE: str = os.getenv("DEPLOYMENT_MODE", "local")  # local or azure

    # Directory paths
    DATA_DIR: Path = Path("./app/data")
    DOCS_DIR: str = "./data/cx_docs"
    VECTOR_DIR: str = "./data/vector_store"
    MEMORY_DIR: str = "./data/memory"
    DOCUMENTS_DIR: str = "./app/data/documents"
    CHUNKED_DIR: str = "./app/data/chunked"
    VECTOR_DB_PATH: str = "app/data/vectorstore"
    PROJECT_DIR: str = "app/data/projects"
    TEMPLATES_DIR: Path = Path("./app/data/templates")
    DEFAULT_CHROMA_COLLECTION: str = "cx_documents"
    UPLOAD_DIR: Path = Path("./app/data/uploads")

    # LLM settings
    LLM_BACKEND: str = Field(
        default="llama.cpp", env="LLM_BACKEND"
    )  # llama.cpp, vllm, ollama, azure
    MODEL_ID: str = Field(default="google/gemma-7b-it", env="MODEL_ID")
    MODEL_PATH: Optional[str] = Field(
        default="models/gemma-2b-it.Q4_K_M.gguf", env="MODEL_PATH"
    )
    MAX_MODEL_LEN: int = Field(default=8192, env="MAX_MODEL_LEN")
    GPU_COUNT: int = Field(default=1, env="GPU_COUNT")
    N_THREADS: Optional[int] = Field(default=None, env="N_THREADS")
    CHAT_FORMAT: str = Field(default="llama-2", env="CHAT_FORMAT")
    LLAMA_CPP_VERBOSE: bool = Field(default=True, env="LLAMA_CPP_VERBOSE")
    LLM_RESPONSE_BUFFER_TOKENS: int = Field(
        default=512, env="LLM_RESPONSE_BUFFER_TOKENS"
    )
    FLASH_ATTENTION: bool = Field(default=False, env="FLASH_ATTENTION")

    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: Optional[str] = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

    # Vector DB settings
    VECTOR_DB_TYPE: str = os.getenv(
        "VECTOR_DB_TYPE", "chroma"
    )  # chroma, azure_ai_search

    # Azure AI Search settings
    AZURE_SEARCH_ENDPOINT: Optional[str] = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_KEY: Optional[str] = os.getenv("AZURE_SEARCH_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "cx-documents")

    # Memory settings
    MEMORY_TYPE: str = os.getenv("MEMORY_TYPE", "redis")  # buffer, redis, azure_redis

    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Azure Redis settings
    AZURE_REDIS_HOST: Optional[str] = os.getenv("AZURE_REDIS_HOST", "")
    AZURE_REDIS_KEY: Optional[str] = os.getenv("AZURE_REDIS_KEY", "")
    AZURE_REDIS_PORT: int = int(os.getenv("AZURE_REDIS_PORT", "6380"))
    AZURE_REDIS_SSL: bool = os.getenv("AZURE_REDIS_SSL", "true").lower() in (
        "true",
        "1",
        "t",
    )

    # Embedding model settings
    EMBEDDING_TYPE: str = "bge"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    BGE_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_THRESHOLD: float = 0.4
    EMBEDDING_DEVICE: str = os.getenv(
        "EMBEDDING_DEVICE", "auto"
    )  # Auto-detect by default

    # ONNX settings
    ONNX_MODEL_PATH: str = "models/bge-small-en-v1.5.onnx"
    ONNX_TOKENIZER: str = "BAAI/bge-small-en-v1.5"

    # Document processing settings
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CHUNK_LENGTH_TOKENS: int = 384
    MAX_DOCUMENTS_PER_QUERY: int = 5
    MAX_CONTEXT_TOKENS_OPTIMIZER: int = 2048

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENABLE_CORS: bool = True
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8080"

    # API settings
    API_PREFIX: str = "/api"

    # Logging settings
    LOG_FILE: str = "app.log"

    # Memory settings
    MAX_MEMORY_ITEMS: int = 10
    TEMPERATURE: float = 0.1
    # PRODUCTION OPTIMIZED: Increased chat history for long conversations
    CHAT_MAX_HISTORY_LENGTH: int = Field(default=1000, env="CHAT_MAX_HISTORY_LENGTH")

    # Project settings
    PROJECT_STORAGE_TYPE: str = "file"
    PROJECT_NAME: str = "CX Consulting AI"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "A description of your project"
    PROJECT_AUTHOR: str = "Your Name"
    PROJECT_AUTHOR_EMAIL: str = "your.email@example.com"
    PROJECT_URL: str = "https://github.com/yourusername/projectname"
    ENABLE_PROJECT_MEMORY: bool = Field(default=False, env="ENABLE_PROJECT_MEMORY")

    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"

    # RAG Configuration & Context Optimization
    CONTEXT_OPTIMIZER_MAX_TOKENS: Optional[int] = None
    CONTEXT_OPTIMIZER_PROMPT_BUFFER: int = 1024
    USE_RERANKING: bool = True
    RERANK_MIN_SCORE_THRESHOLD: float = 0.1

    # Upload settings
    MAX_UPLOAD_SIZE_PER_FILE: int = Field(
        default=100 * 1024 * 1024, env="MAX_UPLOAD_SIZE_PER_FILE"
    )

    # Added for G1 Timeout - Aligned default with typical frontend timeouts (FIX-2)
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "300"))  # Default to 25 seconds

    def get_redis_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection info based on deployment mode."""
        if self.MEMORY_TYPE == "azure_redis":
            return {
                "host": self.AZURE_REDIS_HOST,
                "port": self.AZURE_REDIS_PORT,
                "password": self.AZURE_REDIS_KEY,
                "ssl": self.AZURE_REDIS_SSL,
                "decode_responses": True,
            }
        else:
            return {"url": self.REDIS_URL}

    def is_azure_deployment(self) -> bool:
        """Check if we're using Azure deployment."""
        return self.DEPLOYMENT_MODE == "azure"

    def using_azure_openai(self) -> bool:
        """Check if we're using Azure OpenAI."""
        return self.LLM_BACKEND == "azure"

    def is_production(self) -> bool:
        """Check if we're in production mode."""
        return self.PRODUCTION_MODE or self.DEPLOYMENT_MODE == "azure"

    def get_log_level(self) -> str:
        """Get appropriate log level based on mode."""
        if self.is_production():
            return "WARNING"  # Only warnings and errors in production
        return self.LOG_LEVEL


# @lru_cache(maxsize=1)  # Removing cache to ensure fresh settings
def get_settings() -> Settings:
    return Settings()
