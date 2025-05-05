import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import traceback
import sys
import subprocess
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("cx_consulting_ai")

# Set specific log levels for noisy libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_cpp").setLevel(logging.INFO) # Keep llama_cpp info for now

# Load environment variables
load_dotenv()

# Import application components
from app.core.config import get_settings, Settings
from app.api.routes import router as api_router
from app.api.auth import router as auth_router
from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.auth_logging import AuthLoggingMiddleware
from app.api.dependencies import get_rag_engine, get_memory_manager
from app.utils.redis_manager import ensure_redis_running
from app.core.error_handling import ApplicationError, global_exception_handler

# Import service components
from app.core.llm_service import LLMService
from app.services.document_service import DocumentService
from app.services.context_optimizer import ContextOptimizer
from app.templates.prompt_template import PromptTemplateManager
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.redis_manager import check_redis_status

# Create FastAPI app
app = FastAPI(
    title="CX Consulting AI",
    description="AI assistant for CX consulting tasks",
    version="1.0.0"
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthLoggingMiddleware)

# Configure CORS
if os.getenv("ENABLE_CORS", "true").lower() in ("true", "1", "t"):
    origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
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
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Include API router
app.include_router(api_router)

# Include Auth router
app.include_router(auth_router, prefix="/api/auth")

# Include Deliverables router
from app.api.deliverables import router as deliverables_router
app.include_router(deliverables_router, prefix="/api/deliverables", tags=["Deliverables"])

# Define root endpoint
@app.get("/", tags=["Health"])
async def root():
    """Basic health check."""
    return {"message": "CX Consulting AI is running"}

# Global service instances
service_instances = {}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    logger.info("Starting CX Consulting AI")
    
    start_time = time.time()
    
    try:
        # Check Redis server
        logger.info("Checking Redis server status...")
        redis_running = await check_redis_status()
        if redis_running:
            logger.info("Redis server is running")
        else:
            logger.error("Redis server is not running! Memory features will not work correctly.")
        
        # Initialize LLM Service
        try:
            logger.info(f"Loading LLM service (config read internally)")
            # Instantiate without passing model path/id, let service read from env
            service_instances['llm_service'] = LLMService(
                # model_id=get_settings().MODEL_ID, # REMOVED
                # model_path=get_settings().MODEL_PATH # REMOVED
                # Pass other relevant initial args if needed, like gpu_count
                gpu_count=get_settings().GPU_COUNT,
                max_model_len=get_settings().MAX_MODEL_LEN
            )
            logger.info("LLM service initialized")
        except Exception as e:
            logger.error(f"Error initializing LLM service: {str(e)}")
            traceback.print_exc()
            raise
        
        # Initialize Document Service
        try:
            service_instances['document_service'] = DocumentService(
                documents_dir=os.path.join("app", "data", "documents"),
                chunked_dir=os.path.join("app", "data", "chunked"),
                vectorstore_dir=os.path.join("app", "data", "vectorstore")
            )
            logger.info("Document service initialized")
        except Exception as e:
            logger.error(f"Error initializing Document service: {str(e)}")
            traceback.print_exc()
            raise
        
        # Initialize Template Manager
        try:
            service_instances['template_manager'] = PromptTemplateManager()
            logger.info("Template manager initialized")
        except Exception as e:
            logger.error(f"Error initializing Template manager: {str(e)}")
            traceback.print_exc()
            raise
        
        # Initialize Context Optimizer
        try:
            service_instances['context_optimizer'] = ContextOptimizer(
                max_tokens=get_settings().MAX_CHUNK_LENGTH_TOKENS
            )
            logger.info("Context optimizer initialized")
        except Exception as e:
            logger.error(f"Error initializing Context optimizer: {str(e)}")
            traceback.print_exc()
            # Continue without raising - we'll use a fallback
            service_instances['context_optimizer'] = ContextOptimizer(
                max_tokens=get_settings().MAX_CHUNK_LENGTH_TOKENS,
                use_reranking=False
            )
            logger.warning("Using Context optimizer without reranking due to initialization error")
        
        # Initialize Memory Manager
        try:
            service_instances['memory_manager'] = MemoryManager(
                memory_type=get_settings().MEMORY_TYPE
            )
            logger.info("Memory manager initialized")
        except Exception as e:
            logger.error(f"Error initializing Memory manager: {str(e)}")
            traceback.print_exc()
            # Continue without raising - we'll use a fallback
            service_instances['memory_manager'] = MemoryManager(
                memory_type="buffer"  # Fallback to in-memory buffer
            )
            logger.warning("Using in-memory buffer for Memory manager due to initialization error")
        
        # Initialize RAG Engine
        try:
            service_instances['rag_engine'] = RagEngine(
                llm_service=service_instances['llm_service'],
                document_service=service_instances['document_service'],
                template_manager=service_instances['template_manager'],
                context_optimizer=service_instances['context_optimizer'],
                memory_manager=service_instances['memory_manager']
            )
            logger.info(f"RAG Engine initialized")
        except Exception as e:
            logger.error(f"Error initializing RAG Engine: {str(e)}")
            traceback.print_exc()
            raise
        
        # Store service instances in app state for dependency injection
        app.state.llm_service = service_instances['llm_service']
        app.state.document_service = service_instances['document_service']
        app.state.template_manager = service_instances['template_manager']
        app.state.context_optimizer = service_instances['context_optimizer']
        app.state.memory_manager = service_instances['memory_manager']
        app.state.rag_engine = service_instances['rag_engine']
        
        elapsed_time = time.time() - start_time
        logger.info(f"All services initialized successfully in {elapsed_time:.2f}s")
    
    except ApplicationError as e:
        # Use our custom error handling
        e.log(logging.CRITICAL)
        raise
    except Exception as e:
        # Wrap unknown errors in our ApplicationError
        error = ApplicationError(
            message=f"Error initializing services: {str(e)}",
            original_exception=e
        )
        error.log(logging.CRITICAL)
        raise error 

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CX Consulting AI")
    
    # Clean up resources
    if 'llm_service' in service_instances:
        try:
            # Some LLM backends need explicit cleanup
            service_instances['llm_service'].cleanup()
        except:
            pass 