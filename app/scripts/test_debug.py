#!/usr/bin/env python
"""
Debug Script for CX Consulting AI RAG System

This script tests the RAG components and prints detailed debug information
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import redis
from dotenv import load_dotenv

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.core.llm_service import LLMService
from app.services.chat_service import ChatService
from app.services.context_optimizer import ContextOptimizer
from app.services.document_service import DocumentService
from app.services.rag_engine import RagEngine
from app.template_wrappers.prompt_template import PromptTemplateManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_debug")
logger.setLevel(logging.DEBUG)


def check_redis_running():
    """Check if Redis server is running."""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
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
    llm_service = LLMService()
    print(f"✅ LLM Service initialized with model: {llm_service.model_id}")
    print(f"   Backend: {llm_service.backend}")
    if hasattr(llm_service, "model_path") and llm_service.model_path:
        print(f"   Model path: {llm_service.model_path}")

    # Template Manager
    template_manager = PromptTemplateManager()
    print(
        f"✅ Template Manager initialized with {len(template_manager.templates)} templates"
    )

    # Document Service
    document_service = DocumentService(
        documents_dir=os.getenv("DOCUMENTS_DIR", "app/data/documents"),
        chunked_dir=os.getenv("CHUNKED_DIR", "app/data/chunked"),
        vectorstore_dir=os.getenv("VECTOR_DB_PATH", "app/data/vectorstore"),
        # Add other necessary params like embedding_model, default_collection_name if defaults aren't sufficient
        # embedding_model=os.getenv("EMBEDDING_MODEL"),
        # default_collection_name=os.getenv("DEFAULT_CHROMA_COLLECTION", "cx_documents")
    )
    print(
        f"✅ Document Service initialized with vector db: {document_service.vector_db_type}"
    )

    # Context Optimizer
    context_optimizer = ContextOptimizer()
    print(f"✅ Context Optimizer initialized")

    # Chat Service (Replacing Memory Manager)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    max_history = int(os.getenv("CHAT_MAX_HISTORY_LENGTH", "100"))
    chat_service = ChatService(redis_url=redis_url, max_history_length=max_history)
    print(f"✅ Chat Service initialized with Redis backend, max_history={max_history}")
    # memory_manager = MemoryManager(memory_type="redis") # OLD
    # print(f"✅ Memory Manager initialized with type: {memory_manager.memory_type}") # OLD

    # RAG Engine (No longer takes MemoryManager)
    rag_engine = RagEngine(
        llm_service=llm_service,
        document_service=document_service,
        template_manager=template_manager,
        context_optimizer=context_optimizer,
        # memory_manager=memory_manager, # Remove
        chat_service=chat_service,  # Add
    )
    print(f"✅ RAG Engine initialized")

    # Test document creation
    print("\nCreating test document...")
    test_doc_path = os.path.join("app", "data", "documents", "test_document.txt")
    os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)

    with open(test_doc_path, "w") as f:
        f.write(
            """Customer Experience (CX) refers to the overall perception and feeling that customers have about a brand, based on their interactions across all touchpoints, channels, and products.

Good CX involves understanding customer needs, designing seamless experiences, measuring satisfaction, and continuously improving based on feedback.

Key CX metrics include Net Promoter Score (NPS), Customer Satisfaction (CSAT), Customer Effort Score (CES), and Customer Lifetime Value (CLV).

A CX strategy should focus on customer journey mapping, voice of customer programs, employee experience, digital experience optimization, and continuous measurement."""
        )

    print(f"✅ Test document created at {test_doc_path}")

    # Add document to vector store
    print("\nAdding document to vector store...")
    success = await document_service.add_document(
        document_url=test_doc_path,
        document_type="txt",
        metadata={"source": "test_document", "type": "txt"},
    )

    if success:
        print("✅ Document added to vector store")
    else:
        print("❌ Failed to add document to vector store")
        return

    # Test document retrieval
    print("\nTesting document retrieval...")
    question = "What is customer experience?"
    retrieved_docs = await document_service.retrieve_documents(query=question, limit=5)

    print(f"✅ Retrieved {len(retrieved_docs)} documents for query: '{question}'")
    print("\nRETRIEVED DOCUMENTS:")
    print("====================")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDOCUMENT {i+1}:")
        print(f"Text: {doc['text']}")
        if "metadata" in doc:
            print(f"Metadata: {doc['metadata']}")
        if "score" in doc:
            print(f"Score: {doc['score']}")
        if "distance" in doc:
            print(f"Distance: {doc['distance']}")

    # Test context optimization
    print("\nTesting context optimization...")
    optimized_context = await context_optimizer.optimize(
        question=question, documents=retrieved_docs
    )

    print(f"✅ Context optimized: {len(optimized_context)} characters")
    print("\nOPTIMIZED CONTEXT:")
    print("=================")
    print(optimized_context)

    # --- Test History and Prompt Formatting ---
    print("\nTesting History & Prompt Formatting...")
    # Create a chat for testing history
    test_project_id = "debug-project-001"
    chat_meta = chat_service.create_chat(
        project_id=test_project_id, chat_name="Debug Chat"
    )
    test_chat_id = chat_meta["chat_id"]
    print(f"✅ Created test chat: {test_chat_id}")

    # Add some history
    chat_service.add_message_to_chat(
        test_chat_id, "user", "What is Customer Lifetime Value (CLV)?"
    )
    chat_service.add_message_to_chat(
        test_chat_id,
        "assistant",
        "CLV is a prediction of the net profit attributed to the entire future relationship with a customer.",
    )
    chat_service.add_message_to_chat(test_chat_id, "user", "How is it calculated?")
    print(f"✅ Added history to test chat.")

    # Get history using ChatService
    history_messages = chat_service.get_chat_history(test_chat_id)
    print(f"✅ Retrieved history (oldest first): {len(history_messages)} messages")
    # print(history_messages) # Uncomment to see raw history

    # Format RAG prompt using the new helper (call it directly for debugging)
    print("\nFormatting RAG prompt with history truncation...")
    question = "Give me the formula for CLV calculation."
    context = "Relevant context: CLV = (Average Purchase Value * Purchase Frequency) * Customer Lifespan - Customer Acquisition Cost. Source ID: [doc-metrics-01]"

    # Call the internal helper directly for debug view
    formatted_prompt, history_used = (
        await rag_engine._build_rag_prompt_with_history_truncation(
            question=question,
            context=context,
            chat_history_messages=history_messages,
            project_id=test_project_id,
        )
    )

    print(f"✅ RAG prompt formatted: {len(formatted_prompt)} characters")
    print(f"✅ History messages included in prompt: {len(history_used)}")
    print("\nRAG PROMPT PREVIEW:")
    print("===================")
    print(formatted_prompt[:1000] + "...")  # Print first 1000 chars

    # --- End History Test ---

    # Test LLM generation directly (keep as is, but use the formatted_prompt from above)
    print("\nTesting LLM generation with formatted RAG prompt...")
    start_time = time.time()
    print("Generating response...")
    # Use the prompt built by our helper
    response = await llm_service.generate(prompt=formatted_prompt)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ Generation completed in {elapsed:.2f} seconds")
    print("\nLLM RESPONSE (from RAG prompt):")
    print("===============================")
    print(response)

    # Test full RAG system
    print("\nTesting full RAG system (using ask method on fresh question)...")
    start_time = time.time()

    # Use a new question, the history from test_chat_id will be automatically included by ask()
    rag_question = "What is CLV?"
    rag_response = await rag_engine.ask(
        rag_question, conversation_id=test_chat_id, project_id=test_project_id
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"✅ RAG response generated in {elapsed:.2f} seconds")
    print("\nRAG RESPONSE (ask method):")
    print("=========================")
    print(rag_response)

    # Verify interaction was saved to the correct chat
    final_history = chat_service.get_chat_history(test_chat_id)
    print(f"\nFinal history length for chat {test_chat_id}: {len(final_history)}")
    assert len(final_history) == 5  # 3 added manually + 1 user Q + 1 assistant A
    assert final_history[-2]["content"] == rag_question
    assert final_history[-1]["content"] == rag_response
    print(f"✅ Interaction correctly saved to chat {test_chat_id}")

    # Clean up test files and chat
    os.remove(test_doc_path)
    doc_chunks = os.path.join("app", "data", "chunked", "test_document.json")
    if os.path.exists(doc_chunks):
        os.remove(doc_chunks)
    await chat_service.delete_chat(test_chat_id)
    print(f"✅ Deleted test chat {test_chat_id}")

    print("\n=========== DEBUG COMPLETE ===========\n")


if __name__ == "__main__":
    asyncio.run(debug_rag_system())
