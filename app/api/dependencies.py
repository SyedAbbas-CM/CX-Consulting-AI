# app/api/dependencies.py
import os
import logging
from fastapi import Depends, Request
from functools import lru_cache
from typing import Annotated, Optional

from app.core.config import get_settings, Settings
from app.core.llm_service import LLMService
from app.services.document_service import DocumentService
from app.templates.prompt_template import PromptTemplateManager
from app.services.context_optimizer import ContextOptimizer
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.services.project_manager import ProjectManager
from app.services.chat_service import ChatService

logger = logging.getLogger("cx_consulting_ai.dependencies")

# Global project manager instance (not initialized in startup)
_project_manager: Optional[ProjectManager] = None

# --- RAG Engine Dependency ---
def get_rag_engine(request: Request) -> RagEngine:
    """Dependency to get the initialized RAG Engine instance from app state."""
    service = getattr(request.app.state, 'rag_engine', None)
    if service is None:
        logger.error("RAG Engine not initialized or not found in app state.")
        raise HTTPException(status_code=503, detail="RAG Service Unavailable")
    return service

# --- Memory Manager Dependency ---
def get_memory_manager(request: Request) -> MemoryManager:
    """Dependency to get the initialized Memory Manager instance from app state."""
    service = getattr(request.app.state, 'memory_manager', None)
    if service is None:
        logger.error("Memory Manager not initialized or not found in app state.")
        raise HTTPException(status_code=503, detail="Memory Service Unavailable")
    return service

# --- Project Manager Dependency ---

# Instantiate ProjectManager globally for now (consider if state needs to be request-scoped)
project_manager_instance = ProjectManager()

def get_project_manager() -> ProjectManager:
    """Dependency function to get the ProjectManager instance."""
    # In a more complex setup, this might involve request state or database sessions
    return project_manager_instance

# --- Document Service Dependency ---
# Placeholder for dependency injection (replace with actual DI logic)
document_service_instance: Optional[DocumentService] = None
def get_document_service(request: Request) -> DocumentService:
    """Dependency function to get the DocumentService instance."""
    service = getattr(request.app.state, 'document_service', None)
    if service is None:
        logger.error("Document Service not initialized or not found in app state.")
        raise HTTPException(status_code=503, detail="Document Service Unavailable")
    return service

# --- Authentication Dependency (Placeholder - Implement properly) ---
# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#         token_data = TokenData(username=username)
#     except JWTError:
#         raise credentials_exception
#     user = get_user(fake_users_db, username=token_data.username)
#     if user is None:
#         raise credentials_exception
#     return user

def get_llm_service(request: Request) -> LLMService:
    """Get LLM service from app state."""
    service = getattr(request.app.state, 'llm_service', None)
    if service is None:
        logger.error("LLM Service not initialized or not found in app state.")
        raise HTTPException(status_code=503, detail="LLM Service Unavailable")
    return service

def get_template_manager(request: Request) -> PromptTemplateManager:
    """Get template manager from app state."""
    service = getattr(request.app.state, 'template_manager', None)
    if service is None:
        logger.error("Template Manager not initialized or not found in app state.")
        raise HTTPException(status_code=503, detail="Template Manager Unavailable")
    return service

def get_context_optimizer(request: Request) -> ContextOptimizer:
    """Get context optimizer from app state."""
    service = getattr(request.app.state, 'context_optimizer', None)
    if service is None:
        logger.error("Context Optimizer not initialized or not found in app state.")
        # Allow fallback if ContextOptimizer failed init? Check main.py logic
        raise HTTPException(status_code=503, detail="Context Optimizer Unavailable")
    return service

# --- Chat Service Dependency (Instantiated Globally - Requires Review) ---
# TODO: Review if ChatService should be request-scoped or retrieved from app.state if initialized there
_chat_service_instance: Optional[ChatService] = None
def get_chat_service() -> ChatService:
    """Dependency function to get the ChatService instance."""
    global _chat_service_instance
    if _chat_service_instance is None:
        logger.info("Instantiating ChatService for dependency...")
        try:
            _chat_service_instance = ChatService()
        except Exception as e:
            logger.error(f"Failed to instantiate ChatService: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail="Chat Service unavailable.")
    return _chat_service_instance