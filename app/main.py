import asyncio
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Prometheus integration
from starlette_prometheus import PrometheusMiddleware, metrics

# Silence noisy libraries BEFORE configuring project logger
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log")],
)
logger = logging.getLogger("cx_consulting_ai")

# Set specific log levels for noisy libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_cpp").setLevel(logging.INFO)  # Keep llama_cpp info for now

# Load environment variables
load_dotenv()

import redis  # For redis.exceptions
import redis.asyncio as aredis  # Renamed to aredis to avoid conflict with top-level redis import

from app.api.auth import router as auth_router
from app.api.dependencies import (
    get_chat_service,
    get_context_optimizer,
    get_deliverable_service,
    get_document_service,
    get_llm_service,
    get_project_manager,
    get_project_memory_service,
    get_rag_engine,
    get_template_manager,
    get_template_service,
)
from app.api.middleware.auth_logging import AuthLoggingMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.routes import router as api_router
from app.api.routes_upload import router as upload_router

# Import application components
from app.core.config import Settings, get_settings
from app.core.error_handling import ApplicationError, global_exception_handler

# Import service components
from app.core.llm_service import LLMService
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.context_optimizer import ContextOptimizer
from app.services.deliverable_service import DeliverableService
from app.services.document_service import DocumentService
from app.services.project_manager import ProjectManager
from app.services.project_memory_service import ProjectMemoryService
from app.services.rag_engine import RagEngine
from app.services.template_service import TemplateService
from app.template_wrappers.prompt_template import PromptTemplateManager
from app.utils.redis_manager import ensure_redis_running

# Create FastAPI app
app = FastAPI(
    title="CX Consulting AI",
    description="AI assistant for CX consulting tasks",
    version="1.0.0",
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)
# Add metrics endpoint
app.add_route("/metrics", metrics)

# Add custom Logging middleware (ensure order is considered)
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthLoggingMiddleware)

# Configure CORS
if os.getenv("ENABLE_CORS", "true").lower() in ("true", "1", "t"):
    origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    # Add common localhost ports for development
    if "localhost" in str(origins) or "*" in str(origins):
        origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3002",
            "http://localhost:8000",
            "http://localhost:8080",
            "https://jolly-sand-08d33b71e.6.azurestaticapps.net",
            "*",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled with origins: {origins}")


# Register global exception handler
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# Include API router
app.include_router(api_router)

# Include Auth router
app.include_router(auth_router, prefix="/api/auth")

# Include Upload router
app.include_router(upload_router, prefix="/api", tags=["Document Upload"])

# Include Deliverables router
from app.api.deliverables import router as deliverables_router

app.include_router(
    deliverables_router, prefix="/api/deliverables", tags=["Deliverables"]
)


# Define root endpoint
@app.get("/", tags=["Health"])
async def root():
    """Basic health check."""
    return {"message": "CX Consulting AI is running"}


async def _listen_vectorstore_refresh(app: FastAPI):
    """Background task to listen for vector store refresh events."""
    settings = get_settings()
    redis_client = None  # Initialize
    pubsub = None
    while True:  # Keep trying to connect/reconnect
        try:
            # Ensure DocumentService is initialized before accessing it
            if (
                not hasattr(app.state, "document_service")
                or not app.state.document_service
            ):
                logger.warning(
                    "Vector store refresh listener: DocumentService not ready, waiting..."
                )
                await asyncio.sleep(5)
                continue

            document_service = app.state.document_service

            if redis_client is None:  # Attempt to connect/reconnect
                logger.info("Vector store refresh listener connecting to Redis...")
                redis_client = aredis.Redis.from_url(
                    settings.REDIS_URL, decode_responses=True
                )
                pubsub = redis_client.pubsub()
                await pubsub.subscribe("vectorstore:refresh")
                logger.info(
                    "Vector store refresh listener subscribed to vectorstore:refresh"
                )

            async for message in pubsub.listen():  # This is the generator
                if message is not None and message["type"] == "message":
                    collection_name = message["data"]
                    logger.info(
                        f"Received vectorstore:refresh event for collection: {collection_name}"
                    )
                    try:
                        # Call set_collection with force_reload=True
                        await document_service.set_collection(
                            collection_name, force_reload=True
                        )
                        logger.info(
                            f"Force reloaded collection '{collection_name}' in DocumentService cache."
                        )
                    except Exception as reload_err:
                        logger.error(
                            f"Error reloading collection {collection_name} after refresh event: {reload_err}",
                            exc_info=True,
                        )
        except (
            redis.exceptions.ConnectionError
        ) as redis_conn_err:  # Use top-level redis.exceptions
            logger.error(
                f"Vector store refresh listener: Redis connection error: {redis_conn_err}. Retrying in 10s..."
            )
            if pubsub:
                try:
                    await pubsub.unsubscribe()
                    pubsub.close()  # Close the pubsub instance
                except Exception as ps_close_err:
                    logger.error(f"Error closing pubsub: {ps_close_err}", exc_info=True)
                finally:
                    pubsub = None
            if redis_client:
                try:
                    await redis_client.aclose()  # Use aclose for the async client
                except Exception as rc_close_err:
                    logger.error(
                        f"Error closing redis_client: {rc_close_err}", exc_info=True
                    )
                finally:
                    redis_client = None
            await asyncio.sleep(10)
        except Exception as e:  # Catch other exceptions
            logger.error(
                f"Unexpected error in vector store refresh listener: {e}. Retrying in 15s...",
                exc_info=True,
            )
            if pubsub:
                try:
                    await pubsub.unsubscribe()
                    pubsub.close()
                except Exception as ps_close_err:
                    logger.error(
                        f"Error closing pubsub on generic error: {ps_close_err}",
                        exc_info=True,
                    )
                finally:
                    pubsub = None
            if redis_client:
                try:
                    await redis_client.aclose()
                except Exception as rc_close_err:
                    logger.error(
                        f"Error closing redis_client on generic error: {rc_close_err}",
                        exc_info=True,
                    )
                finally:
                    redis_client = None
            await asyncio.sleep(15)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    settings = get_settings()
    start_time = time.time()
    logger.info(
        "Starting application initialization (true singleton pattern for all services)..."
    )
    logger.info(
        f"SETTINGS CHECK AT STARTUP: EMBEDDING_TYPE='{settings.EMBEDDING_TYPE}', BGE_MODEL_NAME='{settings.BGE_MODEL_NAME}', EMBEDDING_MODEL='{settings.EMBEDDING_MODEL}'"
    )

    try:
        logger.info("Checking Redis server status...")
        await ensure_redis_running(settings.REDIS_URL)

        # --- Instantiate Core Services Directly and Store on app.state ---
        logger.info("Initializing LLMService...")
        app.state.llm_service = LLMService(
            max_model_len=settings.MAX_MODEL_LEN,
        )
        logger.info("LLMService initialized.")

        logger.info("Initializing DocumentService...")

        # Determine the correct embedding model name to pass, based on settings
        initial_embedding_model_name = None
        if settings.EMBEDDING_TYPE.lower() == "bge":
            initial_embedding_model_name = settings.BGE_MODEL_NAME
        else:
            initial_embedding_model_name = settings.EMBEDDING_MODEL

        if not initial_embedding_model_name:
            logger.warning(
                "Could not determine initial embedding model name from settings (EMBEDDING_TYPE, BGE_MODEL_NAME, EMBEDDING_MODEL). DocumentService will attempt to resolve using its internal logic."
            )

        ds = DocumentService(
            documents_dir=str(settings.DOCUMENTS_DIR),
            chunked_dir=str(settings.CHUNKED_DIR),
            vectorstore_dir=str(settings.VECTOR_DB_PATH),
            embedding_model=initial_embedding_model_name,  # Pass the resolved name or None
            default_collection_name=settings.DEFAULT_CHROMA_COLLECTION,
        )
        await ds._init_vector_store()
        app.state.document_service = ds
        logger.info(
            "DocumentService initialized and async _init_vector_store completed."
        )

        logger.info("Initializing PromptTemplateManager...")
        app.state.template_manager = PromptTemplateManager(
            templates_dir=str(settings.TEMPLATES_DIR)
        )
        logger.info("PromptTemplateManager initialized.")

        logger.info("Initializing ContextOptimizer...")
        llm_max_model_len = settings.MAX_MODEL_LEN
        prompt_buffer_tokens = 1024
        optimizer_max_tokens = llm_max_model_len - prompt_buffer_tokens
        if optimizer_max_tokens <= 0:
            optimizer_max_tokens = 1024
        app.state.context_optimizer = ContextOptimizer(
            rerank_model=settings.CROSS_ENCODER_MODEL,
            max_tokens=optimizer_max_tokens,
        )
        logger.info(
            f"ContextOptimizer initialized with max_tokens: {optimizer_max_tokens}"
        )

        logger.info("Initializing ChatService...")
        app.state.chat_service = ChatService(
            redis_url=settings.REDIS_URL,
            max_history_length=settings.CHAT_MAX_HISTORY_LENGTH,
        )
        logger.info("ChatService initialized.")

        logger.info("Initializing ProjectManager...")
        app.state.project_manager = ProjectManager(
            storage_type=settings.PROJECT_STORAGE_TYPE,
            storage_path=settings.PROJECT_DIR,
            document_service=app.state.document_service,
        )
        logger.info("ProjectManager initialized.")

        logger.info("Initializing AuthService (on app.state)...")
        app.state.auth_service = AuthService()
        logger.info("AuthService initialized on app.state.")

        logger.info("Initializing TemplateService...")
        app.state.template_service = TemplateService(
            templates_dir=settings.TEMPLATES_DIR
        )
        logger.info("TemplateService initialized.")

        # --- Instantiate Composite Services using instances from app.state and store them on app.state ---
        logger.info("Initializing ProjectMemoryService (on app.state)...")
        app.state.project_memory_service = ProjectMemoryService(
            llm_service=app.state.llm_service,
            prompt_manager=app.state.template_manager,
            base_dir=settings.PROJECT_DIR,
        )
        logger.info("ProjectMemoryService initialized on app.state.")

        # Initialize RagEngine BEFORE DeliverableService as DeliverableService depends on it
        logger.info("Initializing RagEngine (on app.state)...")
        app.state.rag_engine = RagEngine(
            llm_service=app.state.llm_service,
            document_service=app.state.document_service,
            template_manager=app.state.template_manager,
            context_optimizer=app.state.context_optimizer,
            chat_service=app.state.chat_service,
            # deliverable_service is NOT passed to RagEngine constructor
        )
        logger.info("RagEngine initialized on app.state.")

        logger.info("Initializing DeliverableService (on app.state)...")
        app.state.deliverable_service = DeliverableService(
            llm_service=app.state.llm_service,
            rag_engine=app.state.rag_engine,  # Now app.state.rag_engine exists
            template_service=app.state.template_service,
            memory_service=app.state.project_memory_service,
            chat_service=app.state.chat_service,
        )
        logger.info("DeliverableService initialized on app.state.")

        # Store list of services that might need cleanup
        app.state.services_to_cleanup = [
            app.state.llm_service,
            app.state.document_service,  # Chroma client usually self-manages, but other resources might exist
            app.state.chat_service,  # For Redis client
            app.state.context_optimizer,  # If it holds models like cross-encoder
            app.state.auth_service,  # Add AuthService to cleanup if it has a close/cleanup method
            # Add other services if they have explicit close/cleanup methods
        ]

        elapsed_time = time.time() - start_time
        logger.info(
            f"All services initialized (true singleton on app.state) successfully in {elapsed_time:.2f}s"
        )

        # O3 Fix B-7: Start background listener for vector store refreshes
        logger.info("Starting background task for vector store refresh listener...")
        # Store the task on app.state
        app.state.vectorstore_refresh_task = asyncio.create_task(
            _listen_vectorstore_refresh(app)
        )

    except ApplicationError as e:
        e.log(logging.CRITICAL)
        raise
    except Exception as e:
        error = ApplicationError(
            message=f"Error initializing services: {str(e)} ({type(e).__name__})",
            original_exception=e,
        )
        error.log(logging.CRITICAL)
        raise error


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CX Consulting AI")

    # Cancel the vector store refresh listener task first
    if (
        hasattr(app.state, "vectorstore_refresh_task")
        and app.state.vectorstore_refresh_task
    ):
        logger.info("Cancelling vector store refresh listener task...")
        app.state.vectorstore_refresh_task.cancel()
        try:
            await app.state.vectorstore_refresh_task
        except asyncio.CancelledError:
            logger.info(
                "Vector store refresh listener task was cancelled successfully."
            )
        except Exception as e:
            # Log error if cancellation itself had an issue, but continue shutdown
            logger.error(
                f"Error during cancellation/awaiting of vector store refresh task: {e}",
                exc_info=True,
            )
        finally:
            app.state.vectorstore_refresh_task = None  # Clear from state

    # Iterate over registered services and attempt to close/cleanup
    for service in getattr(app.state, "services_to_cleanup", []):
        service_name = service.__class__.__name__
        try:
            # Prefer async close if available
            if hasattr(service, "aclose") and asyncio.iscoroutinefunction(
                service.aclose
            ):
                logger.info(f"Asynchronously closing {service_name}...")
                await service.aclose()
                logger.info(f"{service_name} closed asynchronously.")
            # Fallback to sync close
            elif hasattr(service, "close") and callable(service.close):
                logger.info(f"Closing {service_name}...")
                # If close might block, consider running in thread for true async shutdown
                # await asyncio.to_thread(service.close)
                service.close()  # Assuming most are quick or manage their own threads
                logger.info(f"{service_name} closed.")
            # Specific cleanup methods if no generic close
            elif hasattr(service, "free_resources") and callable(
                service.free_resources
            ):
                logger.info(f"Freeing resources for {service_name}...")
                service.free_resources()
                logger.info(f"Resources freed for {service_name}.")
            # Add more specific cleanup calls if needed for certain services

        except Exception as e:
            logger.warning(
                f"Error during cleanup of {service_name}: {e}", exc_info=True
            )

    logger.info("Application shutdown complete.")
