#!/usr/bin/env python
"""
Test Script for CX Consulting AI

This script tests that all components of the system are working properly.
"""
import os
import sys
import asyncio
import logging
import subprocess
import shutil
import time
from pathlib import Path
import redis
import platform
from dotenv import load_dotenv

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.core.llm_service import LLMService
from app.templates.prompt_template import PromptTemplateManager
from app.services.document_service import DocumentService
from app.services.context_optimizer import ContextOptimizer
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.utils.redis_manager import ensure_redis_running

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_system")

def check_redis_running():
    """Check if Redis server is running."""
    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        return True
    except (redis.ConnectionError, ConnectionRefusedError):
        return False
    except Exception as e:
        logger.error(f"Error checking Redis: {str(e)}")
        return False

def start_redis_server():
    """Start the Redis server if not running."""
    if check_redis_running():
        logger.info("Redis server is already running")
        return True
    
    try:
        # Check platform to use appropriate command
        if platform.system() == "Windows":
            subprocess.Popen(["redis-server"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        else:
            # macOS or Linux
            subprocess.Popen(["redis-server"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        
        # Wait for Redis to start
        for i in range(5):
            if check_redis_running():
                logger.info("Redis server started successfully")
                return True
            time.sleep(1)
        
        logger.error("Failed to start Redis server")
        return False
    
    except Exception as e:
        logger.error(f"Error starting Redis server: {str(e)}")
        return False

async def test_system():
    """Test all components of the system."""
    logger.info("Testing CX Consulting AI System")
    logger.info("-" * 50)
    
    # Check Redis server
    logger.info("Checking Redis server status...")
    if not check_redis_running():
        logger.error("Redis server is not running. Please start it before running tests.")
        if not start_redis_server():
            return False
    
    # Test LLM Service
    logger.info("Testing LLM Service...")
    try:
        model_id = os.getenv("LLM_MODEL_ID", "google/gemma-7b-it")
        llm_service = LLMService(model_id=model_id)
        logger.info(f"LLM Service initialized with model: {model_id}")
        logger.info(f"Backend: {llm_service.backend}")
        
        prompt = "What is customer experience? Answer in one sentence."
        logger.info(f"Testing generation with prompt: {prompt}")
        
        response_coroutine = llm_service.generate(prompt)
        response = await response_coroutine
        
        token_count = llm_service.count_tokens(prompt)
        logger.info(f"Token count for prompt: {token_count}")
        
        logger.info("LLM Service test successful")
    except Exception as e:
        logger.error(f"LLM Service test failed: {str(e)}")
        return False
    
    logger.info("-" * 50)
    
    # Test Template Manager
    logger.info("Testing Template Manager...")
    try:
        template_manager = PromptTemplateManager()
        logger.info(f"Template Manager initialized with {len(template_manager.templates)} templates")
        
        system_template = template_manager.get_template("system")
        formatted = system_template.format()
        logger.info(f"System template formatted successfully, length: {len(formatted)}")
        
        logger.info("Template Manager test successful")
    except Exception as e:
        logger.error(f"Template Manager test failed: {str(e)}")
        return False
    
    logger.info("-" * 50)
    
    # Test Memory Manager
    logger.info("Testing Memory Manager...")
    try:
        memory_manager = MemoryManager(memory_type="redis")
        logger.info(f"Memory Manager initialized with type: {memory_manager.memory_type}")
        
        # Test adding interaction
        conversation_id = "test-conversation-001"
        await memory_manager.add_interaction(
            conversation_id=conversation_id,
            user_message="Hello, I need help with CX consulting",
            assistant_message="I can help you with various CX consulting tasks such as creating proposals, ROI analyses, and more."
        )
        
        conversation = await memory_manager.get_conversation(conversation_id)
        logger.info(f"Got conversation with {len(conversation)} interactions")
        
        formatted = await memory_manager.format_conversation(conversation_id)
        logger.info(f"Formatted conversation successfully, length: {len(formatted)}")
        
        await memory_manager.clear_conversation(conversation_id)
        conversation = await memory_manager.get_conversation(conversation_id)
        logger.info(f"Cleared conversation, now has {len(conversation)} interactions")
        
        logger.info("Memory Manager test successful")
    except Exception as e:
        logger.error(f"Memory Manager test failed: {str(e)}")
        return False
    
    logger.info("-" * 50)
    
    # Test Context Optimizer
    logger.info("Testing Context Optimizer...")
    try:
        context_optimizer = ContextOptimizer()
        logger.info(f"Context Optimizer initialized with max_tokens: {context_optimizer.max_tokens}")
        
        documents = [
            {
                "text": "Customer Experience (CX) is the sum of all interactions a customer has with a company and its products or services.",
                "metadata": {"source": "test_document_1", "type": "txt"}
            },
            {
                "text": "Net Promoter Score (NPS) is a metric used to measure customer loyalty and satisfaction.",
                "metadata": {"source": "test_document_2", "type": "txt"}
            }
        ]
        
        optimized = await context_optimizer.optimize(
            question="What is NPS?",
            documents=documents
        )
        
        logger.info(f"Optimized context successfully, length: {len(optimized)}")
        logger.info("Context Optimizer test successful")
    except Exception as e:
        logger.error(f"Context Optimizer test failed: {str(e)}")
        return False
    
    logger.info("-" * 50)
    
    # Test Document Service
    logger.info("Testing Document Service...")
    try:
        document_service = DocumentService(
            documents_dir=os.path.join("app", "data", "documents"),
            chunked_dir=os.path.join("app", "data", "chunked"),
            vectorstore_dir=os.path.join("app", "data", "vectorstore")
        )
        logger.info(f"Document Service initialized with vector db: {document_service.vector_db_type}")
        
        test_doc_path = os.path.join("app", "data", "documents", "test_document.txt")
        os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)
        
        with open(test_doc_path, "w") as f:
            f.write("""Customer Experience (CX) refers to the overall perception and feeling that customers have about a brand, based on their interactions across all touchpoints, channels, and products.

Good CX involves understanding customer needs, designing seamless experiences, measuring satisfaction, and continuously improving based on feedback.

Key CX metrics include Net Promoter Score (NPS), Customer Satisfaction (CSAT), Customer Effort Score (CES), and Customer Lifetime Value (CLV).""")
        
        success = await document_service.add_document(
            document_url=test_doc_path,
            document_type="txt",
            metadata={"source": "test_document", "type": "txt"}
        )
        
        if success:
            logger.info("Document added successfully")
        else:
            logger.error("Failed to add document")
            return False
        
        query = "What are CX metrics?"
        results = await document_service.retrieve_documents(query, limit=2)
        
        logger.info(f"Retrieved {len(results)} documents for query: {query}")
        
        os.remove(test_doc_path)
        doc_chunks = os.path.join("app", "data", "chunked", "test_document.json")
        if os.path.exists(doc_chunks):
            os.remove(doc_chunks)
        
        logger.info("Document Service test successful")
    except Exception as e:
        logger.error(f"Document Service test failed: {str(e)}")
        return False
    
    logger.info("-" * 50)
    
    # Test RAG Engine
    logger.info("Testing RAG Engine...")
    try:
        rag_engine = RagEngine(
            llm_service=llm_service,
            document_service=document_service,
            template_manager=template_manager,
            context_optimizer=context_optimizer,
            memory_manager=memory_manager
        )
        logger.info("RAG Engine initialized")
        
        # Test ask method
        question = "What is customer experience?"
        project_id = "test-project-001"
        logger.info(f"Testing ask method with question: {question} for project_id: {project_id}")
        
        try:
            # Get the response from the RAG engine
            response = await rag_engine.ask(question, project_id)
            
            # Ensure the response is not a coroutine
            if asyncio.iscoroutine(response):
                response = await response
                
            logger.info(f"Response received:\n{response}")
            
            # Save the response to a conversation
            if project_id:
                conversation_id = await memory_manager.create_conversation(project_id)
                await memory_manager.add_interaction(
                    conversation_id=conversation_id,
                    user_message=question,
                    assistant_message=response
                )
                logger.info(f"Response saved to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error testing RAG Engine: {e}")
            raise
        
        logger.info("RAG Engine test successful")
    except Exception as e:
        logger.error(f"RAG Engine test failed: {str(e)}")
        return False
    
    logger.info("-" * 50)
    logger.info("All tests completed successfully!")
    return True

if __name__ == "__main__":
    if asyncio.run(test_system()):
        print("\n✅ System tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ System tests failed!")
        sys.exit(1) 