#!/usr/bin/env python
"""
Test Script for RAG Engine

This script specifically tests the RAG engine's ask method with proper await handling.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.core.llm_service import LLMService
from app.templates.prompt_template import PromptTemplateManager
from app.services.document_service import DocumentService
from app.services.context_optimizer import ContextOptimizer
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.utils.redis_manager import ensure_redis_running
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_rag")

async def test_rag_engine():
    """Test the RAG engine's ask method."""
    logger.info("Testing RAG Engine...")
    
    # Ensure Redis is running
    logger.info("Checking Redis server status...")
    ensure_redis_running()
    
    # Initialize services
    llm_service = LLMService()
    template_manager = PromptTemplateManager()
    memory_manager = MemoryManager()
    context_optimizer = ContextOptimizer()
    
    # Properly initialize DocumentService with required parameters
    document_service = DocumentService(
        documents_dir=os.path.join("app", "data", "documents"),
        chunked_dir=os.path.join("app", "data", "chunked"),
        vectorstore_dir=os.path.join("app", "data", "vectorstore")
    )
    
    # Initialize RAG engine
    rag_engine = RagEngine(
        llm_service=llm_service,
        document_service=document_service,
        template_manager=template_manager,
        context_optimizer=context_optimizer,
        memory_manager=memory_manager
    )
    
    # Test ask method
    question = "What is customer experience?"
    project_id = "test-project-001"
    logger.info(f"Testing ask method with question: {question}")
    
    # Properly await the response
    response = await rag_engine.ask(question, project_id=project_id)
    
    # Log the response
    logger.info(f"Response received:\n{response}\n")
    
    logger.info("RAG Engine test completed")
    return True

if __name__ == "__main__":
    asyncio.run(test_rag_engine()) 