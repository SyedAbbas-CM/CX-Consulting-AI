#!/usr/bin/env python
"""
Debug Script for CX Consulting AI RAG System

This script tests the RAG components and prints detailed debug information
"""
import os
import sys
import asyncio
import logging
import time
from pathlib import Path
import redis
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
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_debug")
logger.setLevel(logging.DEBUG)

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

async def debug_rag_system():
    """Debug all components of the RAG system."""
    print("\n========== RAG SYSTEM DEBUG ==========\n")
    
    # Check Redis
    if not check_redis_running():
        print("❌ Redis is not running. Some features may not work correctly.")
    else:
        print("✅ Redis is running")
    
    # Initialize services
    print("\nInitializing services...")
    
    # LLM Service
    model_id = os.getenv("LLM_MODEL_ID", "google/gemma-7b-it")
    llm_service = LLMService(model_id=model_id)
    print(f"✅ LLM Service initialized with model: {model_id}")
    print(f"   Backend: {llm_service.backend}")
    if hasattr(llm_service, 'model_path') and llm_service.model_path:
        print(f"   Model path: {llm_service.model_path}")
    
    # Template Manager
    template_manager = PromptTemplateManager()
    print(f"✅ Template Manager initialized with {len(template_manager.templates)} templates")
    
    # Document Service
    document_service = DocumentService(
        documents_dir=os.path.join("app", "data", "documents"),
        chunked_dir=os.path.join("app", "data", "chunked"),
        vectorstore_dir=os.path.join("app", "data", "vectorstore")
    )
    print(f"✅ Document Service initialized with vector db: {document_service.vector_db_type}")
    
    # Context Optimizer
    context_optimizer = ContextOptimizer()
    print(f"✅ Context Optimizer initialized")
    
    # Memory Manager
    memory_manager = MemoryManager(memory_type="redis")
    print(f"✅ Memory Manager initialized with type: {memory_manager.memory_type}")
    
    # RAG Engine
    rag_engine = RagEngine(
        llm_service=llm_service,
        document_service=document_service,
        template_manager=template_manager,
        context_optimizer=context_optimizer,
        memory_manager=memory_manager
    )
    print(f"✅ RAG Engine initialized")
    
    # Test document creation
    print("\nCreating test document...")
    test_doc_path = os.path.join("app", "data", "documents", "test_document.txt")
    os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)
    
    with open(test_doc_path, "w") as f:
        f.write("""Customer Experience (CX) refers to the overall perception and feeling that customers have about a brand, based on their interactions across all touchpoints, channels, and products.

Good CX involves understanding customer needs, designing seamless experiences, measuring satisfaction, and continuously improving based on feedback.

Key CX metrics include Net Promoter Score (NPS), Customer Satisfaction (CSAT), Customer Effort Score (CES), and Customer Lifetime Value (CLV).

A CX strategy should focus on customer journey mapping, voice of customer programs, employee experience, digital experience optimization, and continuous measurement.""")
    
    print(f"✅ Test document created at {test_doc_path}")
    
    # Add document to vector store
    print("\nAdding document to vector store...")
    success = await document_service.add_document(
        document_url=test_doc_path,
        document_type="txt",
        metadata={"source": "test_document", "type": "txt"}
    )
    
    if success:
        print("✅ Document added to vector store")
    else:
        print("❌ Failed to add document to vector store")
        return
    
    # Test document retrieval
    print("\nTesting document retrieval...")
    question = "What is customer experience?"
    retrieved_docs = await document_service.retrieve_documents(
        query=question,
        limit=5
    )
    
    print(f"✅ Retrieved {len(retrieved_docs)} documents for query: '{question}'")
    print("\nRETRIEVED DOCUMENTS:")
    print("====================")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDOCUMENT {i+1}:")
        print(f"Text: {doc['text']}")
        if 'metadata' in doc:
            print(f"Metadata: {doc['metadata']}")
        if 'score' in doc:
            print(f"Score: {doc['score']}")
        if 'distance' in doc:
            print(f"Distance: {doc['distance']}")
    
    # Test context optimization
    print("\nTesting context optimization...")
    optimized_context = await context_optimizer.optimize(
        question=question,
        documents=retrieved_docs
    )
    
    print(f"✅ Context optimized: {len(optimized_context)} characters")
    print("\nOPTIMIZED CONTEXT:")
    print("=================")
    print(optimized_context)
    
    # Format RAG prompt
    print("\nFormatting RAG prompt...")
    rag_template = template_manager.get_template("rag")
    system_template = template_manager.get_template("system")
    system_prompt = system_template.format()
    
    rag_prompt = rag_template.format(
        system_prompt=system_prompt,
        context=optimized_context,
        conversation_history="",
        query=question
    )
    
    print(f"✅ RAG prompt formatted: {len(rag_prompt)} characters")
    print("\nRAG PROMPT:")
    print("==========")
    print(rag_prompt)
    
    # Test LLM generation directly (with timeout)
    print("\nTesting LLM generation (this may take a minute)...")
    start_time = time.time()
    
    print("Generating response...")
    response_coroutine = llm_service.generate(rag_prompt)
    response = await response_coroutine
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"✅ Generation completed in {elapsed:.2f} seconds")
    print("\nLLM RESPONSE:")
    print("============")
    print(response)
    
    # Test full RAG system
    print("\nTesting full RAG system (using ask method)...")
    start_time = time.time()
    
    rag_response = await rag_engine.ask(question)
    if asyncio.iscoroutine(rag_response):
        rag_response = await rag_response
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"✅ RAG response generated in {elapsed:.2f} seconds")
    print("\nRAG RESPONSE:")
    print("============")
    print(rag_response)
    
    # Clean up test files
    os.remove(test_doc_path)
    doc_chunks = os.path.join("app", "data", "chunked", "test_document.json")
    if os.path.exists(doc_chunks):
        os.remove(doc_chunks)
    
    print("\n=========== DEBUG COMPLETE ===========\n")

if __name__ == "__main__":
    asyncio.run(debug_rag_system()) 