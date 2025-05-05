# app/core/config.py
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

class Settings(BaseModel):
    """Application settings."""
    
    # App settings
    APP_NAME: str = "CX Consulting AI"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")
    
    # Deployment mode
    DEPLOYMENT_MODE: str = os.getenv("DEPLOYMENT_MODE", "local")  # local or azure
    
    # Directory paths
    DOCS_DIR: str = "./data/cx_docs"
    VECTOR_DIR: str = "./data/vector_store"
    MEMORY_DIR: str = "./data/memory"
    DOCUMENTS_DIR: str = "./app/data/documents"
    CHUNKED_DIR: str = "./app/data/chunked"
    
    # LLM settings
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "llama.cpp")  # llama.cpp, vllm, ollama, azure
    MODEL_ID: str = os.getenv("MODEL_ID", "google/gemma-7b-it")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "")
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "8192"))
    GPU_COUNT: int = int(os.getenv("GPU_COUNT", "1"))
    
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: Optional[str] = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    
    # Vector DB settings
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")  # chroma, azure_ai_search
    
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
    AZURE_REDIS_SSL: bool = os.getenv("AZURE_REDIS_SSL", "true").lower() in ("true", "1", "t")
    
    # Embedding model settings
    EMBEDDING_TYPE: str = "bge"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    BGE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # ONNX settings
    ONNX_MODEL_PATH: str = "models/bge-small-en-v1.5.onnx"
    ONNX_TOKENIZER: str = "BAAI/bge-small-en-v1.5"
    
    # Document processing settings
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CHUNK_LENGTH_TOKENS: int = 384
    MAX_DOCUMENTS_PER_QUERY: int = 5
    
    # Vector database settings
    VECTOR_DB_PATH: str = "app/data/vectorstore"
    
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
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Memory settings
    MAX_MEMORY_ITEMS: int = 10
    TEMPERATURE: float = 0.1
    
    # Project settings
    PROJECT_STORAGE_TYPE: str = "file"
    PROJECT_DIR: str = "app/data/projects"
    PROJECT_NAME: str = "CX Consulting AI"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "A description of your project"
    PROJECT_AUTHOR: str = "Your Name"
    PROJECT_AUTHOR_EMAIL: str = "your.email@example.com"
    PROJECT_URL: str = "https://github.com/yourusername/projectname"
    
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"
    CHAT_FORMAT: Optional[str] = "chatml" # Default chat format if using llama.cpp
    N_THREADS: Optional[int] = None # Added for llama.cpp thread count
    
    def get_redis_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection info based on deployment mode."""
        if self.MEMORY_TYPE == "azure_redis":
            return {
                "host": self.AZURE_REDIS_HOST,
                "port": self.AZURE_REDIS_PORT,
                "password": self.AZURE_REDIS_KEY,
                "ssl": self.AZURE_REDIS_SSL,
                "decode_responses": True
            }
        else:
            return {"url": self.REDIS_URL}
    
    def is_azure_deployment(self) -> bool:
        """Check if we're using Azure deployment."""
        return self.DEPLOYMENT_MODE == "azure"
    
    def using_azure_openai(self) -> bool:
        """Check if we're using Azure OpenAI."""
        return self.LLM_BACKEND == "azure"

# Create settings object
settings = Settings()

def get_settings() -> Settings:
    """Get application settings."""
    return settings