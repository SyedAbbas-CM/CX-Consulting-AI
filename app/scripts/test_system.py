#!/usr/bin/env python
"""
Test Script for CX Consulting AI

This script tests that all components of the system are working properly.
"""
import asyncio
import logging
import os
import platform
import shutil
import subprocess
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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_system")


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


def start_redis_server():
    """Start the Redis server if not running."""
    if check_redis_running():
        logger.info("Redis server is already running")
        return True

    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                ["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                ["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        time.sleep(2)  # Give server time to start
        if check_redis_running():
            logger.info("Redis server started successfully")
            return True
        else:
            logger.error("Failed to start Redis server or connect after starting.")
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
        logger.error(
            "Redis server is not running. Please start it before running tests."
        )
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
        logger.info(
            f"Template Manager initialized with {len(template_manager.templates)} templates"
        )

        system_template = template_manager.get_template("system")
        formatted = system_template.format()
        logger.info(f"System template formatted successfully, length: {len(formatted)}")

        logger.info("Template Manager test successful")
    except Exception as e:
        logger.error(f"Template Manager test failed: {str(e)}")
        return False

    logger.info("-" * 50)

    # Test Chat Service
    logger.info("Testing Chat Service...")
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        max_history = int(os.getenv("CHAT_MAX_HISTORY_LENGTH", "100"))
        chat_service = ChatService(redis_url=redis_url, max_history_length=max_history)
        logger.info(
            f"Chat Service initialized with Redis backend, max_history={max_history}"
        )

        project_id = "test-project-for-chat"
        chat_metadata = chat_service.create_chat(
            project_id=project_id, chat_name="System Test Chat"
        )
        assert "chat_id" in chat_metadata
        chat_id = chat_metadata["chat_id"]
        logger.info(f"Created test chat with ID: {chat_id}")

        msg1_ok = chat_service.add_message_to_chat(
            chat_id=chat_id,
            role="user",
            content="Hello, I need help with CX consulting",
        )
        assert msg1_ok
        msg2_ok = chat_service.add_message_to_chat(
            chat_id=chat_id,
            role="assistant",
            content="I can help you with various CX consulting tasks such as creating proposals, ROI analyses, and more.",
        )
        assert msg2_ok
        logger.info("Added messages to chat successfully.")

        history = chat_service.get_chat_history(chat_id)
        logger.info(f"Got conversation history with {len(history)} messages.")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"].startswith("I can help")

        project_chats = chat_service.list_chats_for_project(project_id)
        logger.info(f"Found {len(project_chats)} chats for project {project_id}")
        assert len(project_chats) >= 1
        assert any(c["chat_id"] == chat_id for c in project_chats)

        deleted_ok = await chat_service.delete_chat(chat_id)
        assert deleted_ok
        history_after_delete = chat_service.get_chat_history(chat_id)
        assert len(history_after_delete) == 0
        logger.info(f"Chat {chat_id} deleted successfully.")

        logger.info("Chat Service test successful")
    except Exception as e:
        logger.error(f"Chat Service test failed: {str(e)}", exc_info=True)
        return False

    logger.info("-" * 50)

    # Test Context Optimizer
    logger.info("Testing Context Optimizer...")
    try:
        context_optimizer = ContextOptimizer()
        logger.info(
            f"Context Optimizer initialized with max_tokens: {context_optimizer.max_tokens}"
        )

        documents = [
            {
                "text": "Customer Experience (CX) is the sum of all interactions a customer has with a company and its products or services.",
                "metadata": {"source": "test_document_1", "type": "txt"},
            },
            {
                "text": "Net Promoter Score (NPS) is a metric used to measure customer loyalty and satisfaction.",
                "metadata": {"source": "test_document_2", "type": "txt"},
            },
        ]

        optimized = await context_optimizer.optimize(
            question="What is NPS?", documents=documents
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
            vectorstore_dir=os.path.join("app", "data", "vectorstore"),
        )
        logger.info(
            f"Document Service initialized with vector db: {document_service.vector_db_type}"
        )

        test_doc_path = os.path.join("app", "data", "documents", "test_document.txt")
        os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)

        with open(test_doc_path, "w") as f:
            f.write(
                """Customer Experience (CX) refers to the overall perception and feeling that customers have about a brand, based on their interactions across all touchpoints, channels, and products.

Good CX involves understanding customer needs, designing seamless experiences, measuring satisfaction, and continuously improving based on feedback.

Key CX metrics include Net Promoter Score (NPS), Customer Satisfaction (CSAT), Customer Effort Score (CES), and Customer Lifetime Value (CLV)."""
            )

        success = await document_service.add_document(
            document_url=test_doc_path,
            document_type="txt",
            metadata={"source": "test_document", "type": "txt"},
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
            chat_service=chat_service,
        )
        logger.info("RAG Engine initialized")

        question = "What is customer experience?"
        test_project_id = "test-project-rag-001"
        chat_meta = chat_service.create_chat(
            project_id=test_project_id, chat_name="RAG Test Chat"
        )
        test_chat_id = chat_meta["chat_id"]

        logger.info(
            f"Testing ask method with question: '{question}' for chat_id: {test_chat_id}"
        )

        response = await rag_engine.ask(
            question, conversation_id=test_chat_id, project_id=test_project_id
        )

        assert response and len(response) > 10
        logger.info(f"Response received: {response[:100]}...")

        final_history = chat_service.get_chat_history(test_chat_id)
        assert len(final_history) == 2
        assert final_history[0]["role"] == "user"
        assert final_history[0]["content"] == question
        assert final_history[1]["role"] == "assistant"
        assert final_history[1]["content"] == response
        logger.info(f"RAG response saved to conversation {test_chat_id}")

        await chat_service.delete_chat(test_chat_id)

        logger.info("RAG Engine test successful")
    except Exception as e:
        logger.error(f"RAG Engine test failed: {str(e)}", exc_info=True)
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
