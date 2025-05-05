# app/api/routes.py
from typing import List, Dict, Any, Optional
import time
import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks, status
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import sys
from fastapi import Query
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.api.models import (
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    SearchResult,
    CXStrategyRequest,
    ROIAnalysisRequest,
    JourneyMapRequest,
    DeliverableResponse,
    ConversationInfo,
    ConversationsResponse,
    Project,
    ProjectDocument,
    ProjectCreateRequest,
    ProjectsResponse,
    ProjectDocumentsResponse,
    UserCreate,
    User,
    Token
)
from app.api.projects import ProjectCreate

from app.services.rag_engine import RagEngine
from app.services.memory_manager import MemoryManager
from app.services.project_manager import ProjectManager
from app.api.dependencies import get_rag_engine, get_memory_manager, get_project_manager, get_document_service
from app.api.auth import get_current_user
from app.api.middleware.admin import admin_required
from app.services import auth_service
from app.scripts.model_manager import (
    get_available_models, 
    download_model, 
    update_env_config, 
    check_current_model, 
    MODELS_DIR,
    get_model_status
)
from app.core.llm_service import LLMService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

# Import schemas from their new locations
from app.schemas.model import ModelActionRequest, LlmConfigResponse
from app.schemas.chat import (
    ChatCreateRequest,
    ChatSummaryResponse,
    ChatCreateResponse,
    RefinementResponse,
    ChatHistoryResponse
)

# Import from model_manager script
try:
    # Use absolute import from the app package root
    from app.scripts.model_manager import AVAILABLE_MODELS, download_model, update_env_config, get_available_models, check_current_model, MODELS_DIR, get_model_status 
except ImportError as e:
    logging.error(f"Failed to import from app.scripts.model_manager: {e}")
    # Define placeholders if import fails to avoid runtime errors on router setup
    AVAILABLE_MODELS = {}
    get_available_models = lambda: {}
    check_current_model = lambda: (None, None, None)
    MODELS_DIR = "models"
    get_model_status = lambda model_id: {"status": "error", "message": "model_manager not loaded"}
    def download_model(*args, **kwargs):
        raise NotImplementedError("model_manager.py not found or failed to import")
    def update_env_config(*args, **kwargs):
        raise NotImplementedError("model_manager.py not found or failed to import")

# Configure logger
logger = logging.getLogger("cx_consulting_ai.api.routes")

# Create router
router = APIRouter(prefix="/api")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# --- Instantiate LLMService globally --- 
# This assumes LLMService manages its own state based on .env
# Consider dependency injection patterns for more complex scenarios
try:
    llm_service = LLMService()
except Exception as e:
    logger.error(f"FATAL: Failed to initialize LLMService on startup: {e}", exc_info=True)
    # Decide how to handle this - maybe raise an error to stop FastAPI startup?
    # For now, set to None and endpoints will fail gracefully (or should)
    llm_service = None 

# --- Dependency function to get LLM service (optional, but good practice) ---
def get_llm_service():
    if llm_service is None:
        logger.error("LLM Service was not initialized successfully.")
        raise HTTPException(status_code=503, detail="LLM Service Unavailable")
    return llm_service

# --- Service Instantiation & Dependencies ---
try:
    # Instantiate ChatService globally
    chat_service = ChatService()
except Exception as e:
    logger.error(f"FATAL: Failed to initialize ChatService on startup: {e}", exc_info=True)
    chat_service = None

def get_chat_service():
    """Dependency function to get the ChatService instance."""
    global chat_service # Use the globally instantiated service
    if chat_service is None:
        logger.error("Chat Service was not initialized successfully.")
        raise HTTPException(status_code=503, detail="Chat Service Unavailable")
    return chat_service

@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    is_global: bool = Form(False),
    rag_engine: RagEngine = Depends(get_rag_engine),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Upload a document, checking project access if project_id is provided."""
    logger.info(f"Upload document request for project '{project_id}' by user {current_user.get('id')}")
    try:
        if not project_id and not is_global:
            raise HTTPException(
                status_code=400, 
                detail="Either project_id must be provided or document must be marked as global"
            )
            
        # If project_id is provided, validate it exists AND user has access
        if project_id:
            project = project_manager.get_project(project_id)
            if not project:
                raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
            # Add access check
            if not project_manager.can_access_project(project_id, current_user["id"]):
                 logger.warning(f"User {current_user.get('id')} forbidden to upload document to project {project_id}")
                 raise HTTPException(status_code=403, detail="You do not have permission to upload documents to this project.")
        
        # Check file size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        max_size_mb = 50  # 50 MB max
        
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large: {file_size_mb:.2f} MB. Maximum size: {max_size_mb} MB"
            )
        
        # Process document
        result = await rag_engine.process_document(content, file.filename)
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["error"])
        
        # If project_id is provided, add document to project
        if project_id:
            project_manager.add_document_to_project(project_id, result["document_id"])
        
        # Return successful response
        return DocumentUploadResponse(
            filename=file.filename,
            document_id=result["document_id"],
            project_id=project_id,
            is_global=is_global,
            chunks_created=result["chunks_created"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error uploading document: {str(e)}", exc_info=True) # Added exc_info for better debugging
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    project_manager: ProjectManager = Depends(get_project_manager),
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(get_current_user)
):
    """Ask a question to the CX Consulting AI. Handles conversation creation and message persistence."""
    try:
        start_time = time.time()
            
        conversation_id = request.conversation_id
        project_id = request.project_id
        query = request.query

        # --- Conversation Handling --- 
        if not conversation_id:
            # Create new chat if no ID provided
            logger.info(f"No conversation_id provided. Creating new chat for project '{project_id}'")
            try:
                # Assume project_id must exist if provided for a new chat
                if project_id and not project_manager.get_project(project_id):
                     raise HTTPException(status_code=404, detail=f"Project {project_id} not found for new chat.")
                
                # Use project_id if available, otherwise create an unassociated chat (project_id=None? Needs clarification)
                # For now, let's assume chats MUST belong to a project if project_id is given in the request
                # Or should we always require a project_id for new chats started via /ask?
                # Let's require project_id for now if conversation_id is None.
                if not project_id:
                     raise HTTPException(status_code=400, detail="project_id is required when starting a new chat via /ask")

                new_chat_info = chat_service.create_chat(project_id=project_id) # Use default name
                conversation_id = new_chat_info['chat_id']
                logger.info(f"Created new chat with id: {conversation_id} for project {project_id}")
            except Exception as create_err:
                 logger.error(f"Failed to create new chat: {create_err}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Could not start a new chat session: {create_err}")
        else:
            # Verify existing chat and project association
            try:
                 chats = chat_service.list_chats_for_project(project_id=project_id, limit=1000) if project_id else []
                 chat_exists = any(chat['chat_id'] == conversation_id for chat in chats)
                 
                 if project_id and not chat_exists:
                      # Check if chat exists but belongs to a different project? (More complex check needed in ChatService maybe)
                      logger.warning(f"Chat {conversation_id} not found within project {project_id}.")
                      # We could try to just fetch the chat directly to see if it exists at all
                      # history_check = chat_service.get_chat_history(conversation_id, limit=1) # Efficient check? Requires get_chat_history to handle not found
                      # For now, raise error if project association seems wrong
                      raise HTTPException(status_code=404, detail=f"Chat {conversation_id} not found or does not belong to project {project_id}")
                 elif not project_id:
                      # If no project specified, ensure chat exists globally (how?) - needs more thought.
                      # For now, assume if project_id is null, we don't validate association strictly.
                      pass # Allow proceeding if no project context is given?
            except Exception as check_err:
                 logger.error(f"Error verifying chat {conversation_id} / project {project_id}: {check_err}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Error verifying chat session: {check_err}")
        # --- End Conversation Handling ---
        
        # Add user message to persistent chat history
        try:
            added_user_msg = chat_service.add_message_to_chat(
                chat_id=conversation_id,
                role="user",
                content=query
            )
            if not added_user_msg:
                logger.error(f"Failed to add user message to chat {conversation_id}")
                # Raise an error? Or just log and continue?
                raise HTTPException(status_code=500, detail="Failed to save user message to chat history")
        except Exception as e:
             logger.error(f"Exception adding user message to chat {conversation_id}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Error saving user message")

        # Generate answer using the RAG engine (still uses MemoryManager internally for context window)
        response = await rag_engine.ask(
            question=query,
            conversation_id=conversation_id, # RAG engine uses this for MemoryManager context
            project_id=project_id 
        )
        
        # Add assistant message to persistent chat history
        try:
            added_assistant_msg = chat_service.add_message_to_chat(
                chat_id=conversation_id,
                role="assistant",
                content=response
            )
            if not added_assistant_msg:
                 logger.error(f"Failed to add assistant message to chat {conversation_id}")
                 # Non-critical? Log warning and return response anyway?
                 # Let's make it critical for now
                 raise HTTPException(status_code=500, detail="Failed to save assistant response to chat history")
        except Exception as e:
             logger.error(f"Exception adding assistant message to chat {conversation_id}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Error saving assistant response")

        # Prepare sources (RAG engine needs to return this)
        # TODO: Modify rag_engine.ask to return sources along with the answer
        sources = [] 
        
        processing_time = time.time() - start_time
            
        logger.info(f"Successfully processed /api/ask in {processing_time:.4f}s for chat {conversation_id}")
        
        return QuestionResponse(
            answer=response,
            conversation_id=conversation_id,
            project_id=project_id, # Return the project_id used
            sources=sources, # TODO: Send actual sources
            processing_time=processing_time
        )

    except HTTPException:
        raise # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.error(f"Error processing /api/ask: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@router.post("/cx-strategy", response_model=DeliverableResponse)
async def generate_cx_strategy(
    request: CXStrategyRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    # memory_manager: MemoryManager = Depends(get_memory_manager), # MemoryManager is now internal to RagEngine
    project_manager: ProjectManager = Depends(get_project_manager), # Still needed for access checks
    current_user: dict = Depends(get_current_user)
):
    """Generate a CX strategy document using the v2 template."""
    logger.info(f"CX Strategy request for project {request.project_id} by user {current_user.get('id')}")
    try:
        start_time = time.time()
        project_id = request.project_id
        conversation_id = request.conversation_id # Can be None

        # Check project access
        if project_id and not project_manager.can_access_project(project_id, current_user["id"]):
             raise HTTPException(status_code=403, detail="You do not have permission to access this project.")

        # Convert request model to dict for RagEngine method
        request_data = request.model_dump()

        # Call the new RagEngine method
        response_content, updated_conversation_id, tokens = await rag_engine.create_cx_strategy_from_template(
            request_data=request_data,
            conversation_id=conversation_id
        )
        
        # TODO: Consider if/how to store the generated deliverable as a document
        document_id = None 
    
        processing_time = time.time() - start_time
    
        return DeliverableResponse(
            content=response_content,
            conversation_id=updated_conversation_id,
            project_id=project_id,
            document_id=document_id,
            processing_time=processing_time
        )
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error generating CX strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating CX strategy: {str(e)}")

@router.post("/roi-analysis", response_model=DeliverableResponse)
async def generate_roi_analysis(
    request: ROIAnalysisRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Generate an ROI analysis using the v2 template."""
    logger.info(f"ROI Analysis request for project {request.project_id} by user {current_user.get('id')}")
    try:
        start_time = time.time()
        project_id = request.project_id
        conversation_id = request.conversation_id
        
        # Check project access
        if project_id and not project_manager.can_access_project(project_id, current_user["id"]):
             raise HTTPException(status_code=403, detail="You do not have permission to access this project.")

        request_data = request.model_dump()

        # Call the new RagEngine method
        response_content, updated_conversation_id, tokens = await rag_engine.create_roi_analysis_from_template(
            request_data=request_data,
            conversation_id=conversation_id
        )
        
        document_id = None
    
        processing_time = time.time() - start_time
    
        return DeliverableResponse(
            content=response_content,
            conversation_id=updated_conversation_id,
            project_id=project_id,
            document_id=document_id,
            processing_time=processing_time
        )
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error generating ROI analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating ROI analysis: {str(e)}")

@router.post("/journey-map", response_model=DeliverableResponse)
async def generate_journey_map(
    request: JourneyMapRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Generate a customer journey map using the v2 template."""
    logger.info(f"Journey Map request for project {request.project_id} by user {current_user.get('id')}")
    try:
        start_time = time.time()
        project_id = request.project_id
        conversation_id = request.conversation_id
        
        # Check project access
        if project_id and not project_manager.can_access_project(project_id, current_user["id"]):
             raise HTTPException(status_code=403, detail="You do not have permission to access this project.")

        request_data = request.model_dump()

        # Call the new RagEngine method
        response_content, updated_conversation_id, tokens = await rag_engine.create_journey_map_from_template(
            request_data=request_data,
            conversation_id=conversation_id
        )
        
        document_id = None
    
        processing_time = time.time() - start_time
    
        return DeliverableResponse(
            content=response_content,
            conversation_id=updated_conversation_id,
            project_id=project_id,
            document_id=document_id,
            processing_time=processing_time
        )
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error generating journey map: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating journey map: {str(e)}")

@router.get("/conversations", response_model=ConversationsResponse)
async def get_conversations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    project_id: Optional[str] = None,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get all conversations."""
    try:
        # If project_id is provided, get conversations for that project
        if project_id:
            conversation_ids = memory_manager.get_project_conversations(project_id)
            conversation_list = []
            
            for conv_id in conversation_ids:
                conv = memory_manager.get_conversation(conv_id)
                if conv:
                    # Calculate timestamps
                    created_at = datetime.fromtimestamp(conv[0]["timestamp"]).isoformat() if conv else datetime.now().isoformat()
                    updated_at = datetime.fromtimestamp(conv[-1]["timestamp"]).isoformat() if conv else datetime.now().isoformat()
                    
                    conversation_list.append(ConversationInfo(
                        id=conv_id,
                        created_at=created_at,
                        updated_at=updated_at,
                        message_count=len(conv),
                        project_id=project_id
                    ))
            
            # Apply pagination
            paginated = conversation_list[offset:offset+limit]
            return ConversationsResponse(
                conversations=paginated,
                count=len(conversation_list)
            )
        else:
            # For now, just return an empty list - this would need to be updated
            # to get all conversations in a real implementation
            return ConversationsResponse(
                conversations=[],
                count=0
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversations: {str(e)}")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation."""
    try:
        # Check if conversation is associated with a project
        project_id = memory_manager.get_conversation_project(conversation_id)
        if project_id:
            # Remove from project's conversation list
            project = project_manager.get_project(project_id)
            if project and conversation_id in project.get("conversation_ids", []):
                project_manager.remove_conversation_from_project(project_id, conversation_id)
        
        # Delete conversation
        success = memory_manager.delete_conversation(conversation_id)
        if success:
            return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

@router.get("/health")
async def health_check(
    memory_manager: MemoryManager = Depends(get_memory_manager),
    rag_engine: RagEngine = Depends(get_rag_engine)
):
    """Health check endpoint that verifies the status of all system components."""
    import time
    from app.utils.redis_manager import check_redis_running
    
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "model": os.getenv("MODEL_ID", "google/gemma-7b-it"),
        "version": "1.0.0",
        "services": {}
    }
    
    # Check Redis
    redis_running = check_redis_running()
    health_status["services"]["redis"] = {
        "status": "healthy" if redis_running else "unhealthy",
        "details": "Connected" if redis_running else "Connection failed"
    }
    
    # Check Memory Manager
    try:
        test_key = "health_check_test"
        memory_manager.redis_client.set(test_key, "test_value")
        memory_manager.redis_client.delete(test_key)
        health_status["services"]["memory_manager"] = {
            "status": "healthy",
            "details": "Redis operations successful"
        }
    except Exception as e:
        health_status["services"]["memory_manager"] = {
            "status": "unhealthy",
            "details": f"Redis operations failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check Vector DB
    try:
        vector_db = rag_engine.document_service.vector_store
        collection_name = rag_engine.document_service.collection_name
        collection = vector_db.get_collection(name=collection_name)
        if collection is not None:
            health_status["services"]["vector_db"] = {
                "status": "healthy",
                "details": f"Connected to collection: {collection_name}"
            }
        else:
            health_status["services"]["vector_db"] = {
                "status": "degraded",
                "details": f"Collection not found: {collection_name}"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["vector_db"] = {
            "status": "unhealthy",
            "details": f"Vector DB operations failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check LLM Service
    try:
        llm_service = rag_engine.llm_service
        token_count = llm_service.count_tokens("test")
        if token_count > 0:
            health_status["services"]["llm_service"] = {
                "status": "healthy",
                "details": f"Model loaded: {llm_service.model_id}"
            }
        else:
            health_status["services"]["llm_service"] = {
                "status": "degraded",
                "details": "Tokenizer issue detected"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["llm_service"] = {
            "status": "unhealthy",
            "details": f"LLM service operations failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Add response time
    health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return health_status

# PROJECTS API ENDPOINTS

@router.post("/projects", response_model=Project)
async def create_project(
    project_request: ProjectCreateRequest,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Creates a new project."""
    try:
        # Assume create_project returns the ID of the newly created project
        new_project_id = project_manager.create_project(
            name=project_request.name,
            client_name=project_request.client_name,
            industry=project_request.industry,
            description=project_request.description,
            owner_id=current_user["id"],
            shared_with=project_request.shared_with,
            metadata=project_request.metadata
        )
        
        # Fetch the full project details using the ID to match the response_model
        project = project_manager.get_project(new_project_id)
        if not project:
             # This shouldn't happen if creation succeeded, but handle defensively
             logger.error(f"Failed to fetch project details immediately after creation for ID: {new_project_id}")
             raise HTTPException(status_code=404, detail="Failed to retrieve created project details.")
             
        return project # Return the full Project object
    except Exception as e:
        logger.error(f"Unexpected error creating project '{project_request.name}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the project.")

@router.get("/projects", response_model=ProjectsResponse)
async def list_projects(
    # Remove limit/offset as filtering happens on the full user list now
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """List all projects accessible by the current user (owned or shared)."""
    try:
        # Fetch only projects accessible by the current user
        projects = project_manager.get_user_projects(current_user["id"], include_shared=True)
        # Count is simply the length of the accessible projects list
        count = len(projects)
        # Note: Pagination would need to be applied here if desired, after fetching.
        # For simplicity now, return all accessible projects.
        return ProjectsResponse(projects=projects, count=count)
    except Exception as e:
        logger.error(f"Error listing projects for user {current_user.get('id')}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")

@router.get("/projects/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get a project by ID, checking access."""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            # Keep 404 if project doesn't exist at all
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Add access check
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden access to project {project_id}")
             raise HTTPException(status_code=403, detail="You do not have permission to access this project.")
             
        return project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project {project_id} for user {current_user.get('id')}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting project: {str(e)}")

@router.put("/projects/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    updates: ProjectCreateRequest, # Use the Pydantic model for updates
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Update a project. Only the owner can update."""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

        # Check ownership
        if project.get("owner_id") != current_user.get("id"):
            raise HTTPException(status_code=403, detail="Only the project owner can update the project.")

        # Convert Pydantic model to dict, excluding unset fields to avoid overwriting with None
        update_data = updates.model_dump(exclude_unset=True)

        success = project_manager.update_project(project_id, update_data)
        if not success:
            # This might indicate a race condition or internal error if the project existed moments ago
            raise HTTPException(status_code=500, detail=f"Failed to update project {project_id}")

        # Return the updated project
        updated_project = project_manager.get_project(project_id)
        if not updated_project:
             raise HTTPException(status_code=404, detail=f"Updated project {project_id} could not be retrieved.")
        return updated_project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating project: {str(e)}")

@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
# Remove @admin_required, check ownership inside
async def delete_project(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    chat_service: ChatService = Depends(get_chat_service), # Need chat service to delete associated chats
    current_user: dict = Depends(get_current_user)
):
    """Delete a project and its associated chats. Only the owner can delete."""
    logger.info(f"Attempting to delete project {project_id} by user {current_user.get('id')}")
    try:
        # 1. Check if project exists
        project = project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for deletion.")
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

        # 2. Check ownership
        if project.get("owner_id") != current_user.get("id"):
            logger.warning(f"User {current_user.get('id')} attempted to delete project {project_id} owned by {project.get('owner_id')}")
            raise HTTPException(status_code=403, detail="Only the project owner can delete the project.")

        # 3. Get associated chat IDs BEFORE deleting the project
        # Assuming project_manager stores chat IDs, or we need to query chat_service
        # Let's assume ProjectManager holds the list
        chat_ids_to_delete = project.get("conversation_ids", []) # Use 'conversation_ids' as per Project model
        logger.info(f"Found {len(chat_ids_to_delete)} chats associated with project {project_id} for deletion: {chat_ids_to_delete}")

        # 4. Delete associated chats
        deleted_chat_count = 0
        failed_chat_deletions = []
        for chat_id in chat_ids_to_delete:
            try:
                # Use the new delete_chat endpoint logic (or call chat_service directly)
                success = await chat_service.delete_chat(chat_id)
                if success:
                    deleted_chat_count += 1
                    logger.info(f"Deleted associated chat {chat_id} for project {project_id}")
                else:
                    # This might happen if the chat was already deleted or doesn't exist
                    logger.warning(f"Could not delete associated chat {chat_id} for project {project_id} (might not exist).")
                    failed_chat_deletions.append(chat_id)
            except Exception as chat_del_err:
                logger.error(f"Error deleting associated chat {chat_id} for project {project_id}: {chat_del_err}", exc_info=True)
                failed_chat_deletions.append(chat_id)

        if failed_chat_deletions:
             # Decide how to handle partial failures. For now, log a warning and proceed with project deletion.
             logger.warning(f"Failed to delete some associated chats for project {project_id}: {failed_chat_deletions}")


        # 5. Delete the project itself
        success = project_manager.delete_project(project_id)
        if not success:
            # Should not happen if checks passed, but handle defensively
            logger.error(f"Project manager failed to delete project {project_id} after ownership check.")
            raise HTTPException(status_code=500, detail=f"Failed to delete project {project_id} after deleting chats.")

        logger.info(f"Successfully deleted project {project_id} and {deleted_chat_count} associated chats.")
        return None # Return None for 204 No Content
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")

# PROJECT DOCUMENTS API ENDPOINTS

@router.get("/projects/{project_id}/documents", response_model=ProjectDocumentsResponse)
async def get_project_documents(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get all documents for a project, checking access."""
    logger.info(f"Request for documents in project {project_id} by user {current_user.get('id')}")
    try:
        # Check if project exists and user has access
        # Use can_access_project which implicitly checks existence via get_project
        if not project_manager.can_access_project(project_id, current_user["id"]):
             # Distinguish between not found and forbidden
             project_exists = project_manager.get_project(project_id) is not None
             if not project_exists:
                  logger.warning(f"Project {project_id} not found when listing documents.")
                  raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
             else:
                  logger.warning(f"User {current_user.get('id')} forbidden to list documents for project {project_id}")
                  raise HTTPException(status_code=403, detail="You do not have permission to access documents in this project.")
        
        # If access is granted, get the documents
        documents = project_manager.get_project_documents(project_id)
        return ProjectDocumentsResponse(documents=documents, count=len(documents))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting documents for project {project_id}, user {current_user.get('id')}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting project documents: {str(e)}")

@router.get("/documents/{document_id}", response_model=ProjectDocument)
async def get_document(
    document_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get a document by ID, checking access via its project."""
    logger.info(f"Request for document {document_id} by user {current_user.get('id')}")
    try:
        document = project_manager.get_document(document_id)
        if not document or not document.get("project_id"):
            logger.warning(f"Document {document_id} not found or missing project_id.")
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found or invalid.")
        
        project_id = document["project_id"]
        
        # Check access to the project this document belongs to
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden to access document {document_id} (project {project_id}).")
             raise HTTPException(status_code=403, detail="You do not have permission to access this document.")
             
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id} for user {current_user.get('id')}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@router.put("/documents/{document_id}", response_model=ProjectDocument)
async def update_document(
    document_id: str,
    updates: dict,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Update a document, checking access via its project."""
    logger.info(f"Request to update document {document_id} by user {current_user.get('id')}")
    try:
        # 1. Get document and check existence
        document = project_manager.get_document(document_id)
        if not document or not document.get("project_id"):
            logger.warning(f"Document {document_id} not found or missing project_id for update.")
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found or invalid.")
            
        project_id = document["project_id"]
        
        # 2. Check access to the project
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden to update document {document_id} (project {project_id}).")
             raise HTTPException(status_code=403, detail="You do not have permission to update this document.")

        # 3. Perform update 
        success = project_manager.update_document(document_id, updates)
        if not success:
            # This might happen if the document was deleted between check and update
            logger.error(f"Failed to update document {document_id} after access check.")
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found during update.")
        
        # Return the updated document
        updated_document = project_manager.get_document(document_id)
        if not updated_document:
            # Should not happen if update was successful
            logger.error(f"Failed to retrieve document {document_id} after successful update.")
            raise HTTPException(status_code=500, detail="Failed to retrieve document after update.")
        return updated_document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id} for user {current_user.get('id')}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")

@router.delete("/documents/{document_id}",
               status_code=status.HTTP_204_NO_CONTENT, 
               tags=["Documents"], # Add to Documents tag group
               summary="Delete a document and its chunks, checking access")
async def delete_document_route(
    document_id: str,
    doc_service: DocumentService = Depends(get_document_service),
    project_manager: ProjectManager = Depends(get_project_manager), # Need ProjectManager for access check
    current_user: dict = Depends(get_current_user) # Secure endpoint
):
    """
    Deletes a document and its associated chunks from the vector store,
    after verifying the user has access to the document's project.
    """
    logger.info(f"Received request to delete document {document_id} by user {current_user.get('id')}")
    try:
        # 1. Get document details to find project ID
        document = project_manager.get_document(document_id)
        if not document or not document.get("project_id"):
            # If document doesn't exist, treat as success (idempotent delete)
            logger.warning(f"Document {document_id} not found or invalid for deletion. Assuming already deleted.")
            return None # Return 204 No Content
            
        project_id = document["project_id"]
        
        # 2. Check access to the project
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden to delete document {document_id} (project {project_id}).")
             raise HTTPException(status_code=403, detail="You do not have permission to delete this document.")

        # 3. Perform the deletion using DocumentService (vector store)
        success = await doc_service.delete_document(document_id)
        if not success:
            # Log error but don't necessarily fail if vector store deletion had issues,
            # as we still want to remove it from the project manager.
            logger.error(f"Document service failed to process vector store deletion for ID: {document_id}")
            # Raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            #                     detail=f"Failed to delete document '{document_id}' chunks. Check server logs.")

        # 4. Remove document reference from ProjectManager (even if vector deletion failed)
        # Use ProjectManager's delete method which handles file/redis removal and project list update
        pm_delete_success = project_manager.delete_document(document_id)
        if not pm_delete_success:
             # This shouldn't happen if get_document worked, but log defensively
             logger.error(f"Project manager failed to remove reference for document {document_id} after access check.")
             # Consider if this should be a 500 error

        logger.info(f"Successfully processed deletion request for document ID: {document_id} (Project Manager reference removed).")
        return None # Return None for 204 response

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in delete document route for ID {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred while deleting document '{document_id}'.")

# PROJECT CONVERSATIONS API ENDPOINTS

@router.get("/projects/{project_id}/conversations", response_model=ConversationsResponse)
async def get_project_conversations(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get all conversations for a project."""
    try:
        # Check if project exists
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Get conversation IDs
        conversation_ids = memory_manager.get_project_conversations(project_id)
        
        # Get conversation details
        conversations = []
        for conv_id in conversation_ids:
            conv = memory_manager.get_conversation(conv_id)
            if conv:
                # Create a conversation info object
                conversations.append(ConversationInfo(
                    id=conv_id,
                    created_at=datetime.fromtimestamp(conv[0]["timestamp"]).isoformat() if conv else datetime.now().isoformat(),
                    updated_at=datetime.fromtimestamp(conv[-1]["timestamp"]).isoformat() if conv else datetime.now().isoformat(),
                    message_count=len(conv),
                    project_id=project_id
                ))
        
        return ConversationsResponse(conversations=conversations, count=len(conversations))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting project conversations: {str(e)}")

@router.post("/projects/{project_id}/associate-conversation/{conversation_id}")
async def associate_conversation_with_project(
    project_id: str,
    conversation_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: dict = Depends(get_current_user)
):
    """Associate a conversation with a project."""
    try:
        # Check if project exists
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Check if conversation exists
        conversation = memory_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        # Associate conversation with project
        memory_manager.set_conversation_project(conversation_id, project_id)
        project_manager.add_conversation_to_project(project_id, conversation_id)
        
        return {"status": "success", "message": f"Conversation {conversation_id} associated with project {project_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error associating conversation with project: {str(e)}")

# ADMIN ENDPOINTS

@router.get("/admin/users")
@admin_required
async def list_users(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """List all users (admin only)."""
    try:
        from app.services.auth_service import AuthService
        auth_service = AuthService()
        
        users, count = auth_service.list_users(limit=limit, offset=offset)
        return {"users": users, "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing users: {str(e)}")

@router.put("/admin/users/{user_id}")
@admin_required
async def admin_update_user(
    user_id: str,
    updates: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update a user (admin only)."""
    try:
        from app.services.auth_service import AuthService
        auth_service = AuthService()
        
        # Don't allow changing password through this endpoint
        if "password" in updates:
            del updates["password"]
        
        user = auth_service.update_user(user_id, updates)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")

@router.delete("/admin/users/{user_id}")
@admin_required
async def admin_delete_user(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a user (admin only)."""
    try:
        from app.services.auth_service import AuthService
        auth_service = AuthService()
        
        # Don't allow deleting yourself
        if user_id == current_user["id"]:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
        
        success = auth_service.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return {"status": "success", "message": f"User {user_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@router.post("/admin/users")
@admin_required
async def admin_create_user(
    user_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create a new user (admin only)."""
    try:
        from app.services.auth_service import AuthService
        auth_service = AuthService()
        
        # Check required fields
        required_fields = ["username", "email", "password"]
        for field in required_fields:
            if field not in user_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create user
        user = auth_service.create_user(
            username=user_data["username"],
            email=user_data["email"],
            password=user_data["password"],
            full_name=user_data.get("full_name"),
            company=user_data.get("company"),
            is_admin=user_data.get("is_admin", False)
        )
        
        if not user:
            raise HTTPException(
                status_code=400,
                detail="Could not create user. Username or email may already exist."
            )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.get("/improvement/interactions", response_model=Dict[str, List[Dict[str, Any]]])
async def get_improvement_interactions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Get saved interactions for model improvement."""
    try:
        # Only admins can access this endpoint
        if not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to access this resource"
            )
        
        # Get the improvement directory
        improvement_dir = os.path.join("app", "data", "improvement")
        
        # Create directory if it doesn't exist
        os.makedirs(improvement_dir, exist_ok=True)
        
        # Get all JSON files in the directory
        files = sorted(
            glob.glob(os.path.join(improvement_dir, "*.json")),
            key=os.path.getmtime,
            reverse=True
        )
        
        # Apply pagination
        paginated_files = files[offset:offset + limit]
        
        # Load and return the interactions
        interactions = []
        for file_path in paginated_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Add file ID based on filename
                    data["id"] = Path(file_path).stem
                    interactions.append(data)
            except Exception as e:
                logger.warning(f"Error loading interaction file {file_path}: {str(e)}")
                
        return {"interactions": interactions, "total": len(files)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting improvement interactions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving interactions: {str(e)}")

# --- NEW Model Management Routes --- 

@router.get("/models", tags=["Models"])
async def list_available_models(current_user: dict = Depends(get_current_user)):
    """Lists available models and their download status."""
    models_status_list = []
    # Use the imported function that handles potential import errors
    available_models_dict = get_available_models() 
    
    if not available_models_dict: 
        logger.warning("AVAILABLE_MODELS dictionary is empty or model_manager failed to load.")
        # Return empty list or appropriate error response?
        # Let's return empty for now, but ideally we'd know if it failed vs. was just empty.
        return {"available_models": [], "active_model_path": "Unknown"}
        
    try:
        for model_id, info in available_models_dict.items():
            # Get status for each model
            status_info = get_model_status(model_id)
            
            models_status_list.append({
                # Base info from AVAILABLE_MODELS
                "id": model_id,
                "name": info.get("description", model_id), # Use description as name, fallback to id
                "description": info.get("description"),
                "size_gb": info.get("size_gb"),
                # Status info from get_model_status
                "status": status_info.get("status", "unknown"),
                "message": status_info.get("message", ""),
                "path": status_info.get("path"), # Include path if available
                # Add a simple boolean for frontend convenience
                "downloaded": status_info.get("status") == "available" 
            })
            
        # Get currently configured model path from .env (using check_current_model)
        _, active_filename, _ = check_current_model()
        active_model_path = os.path.join(MODELS_DIR, active_filename) if active_filename else "Unknown"
            
        return {"available_models": models_status_list, "active_model_path": active_model_path}
    except Exception as e:
        logger.error(f"Error listing models with status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models")

@router.post("/models/download", tags=["Models"])
async def download_model_route(request: ModelActionRequest, background_tasks: BackgroundTasks):
    """Triggers a model download in the background."""
    model_id = request.model_id
    force = request.force_download
    logger.info(f"Received request to download model: {model_id}, Force: {force}")
    
    available_models = get_available_models()
    if model_id not in available_models:
        logger.warning(f"Download request for unknown model_id: {model_id}")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in available models.")

    # Simple check: If model is already available, don't re-download unless forced
    status_info = get_model_status(model_id)
    if status_info["status"] == "available" and not force:
         logger.info(f"Model '{model_id}' already available. No download triggered.")
         return {"message": f"Model '{model_id}' is already downloaded."}
    if status_info["status"] == "downloading":
         logger.info(f"Model '{model_id}' is already downloading.")
         return {"message": f"Model '{model_id}' download already in progress."}

    # Run the download in the background
    logger.info(f"Adding download task for '{model_id}' to background tasks.")
    background_tasks.add_task(download_model, model_id, force=force)
    
    return {"message": f"Download started for model '{model_id}'. Check status endpoint for progress."}

@router.post("/models/set_active", tags=["Models"])
async def set_active_model_route(request: ModelActionRequest, current_llm_service: LLMService = Depends(get_llm_service)):
    """Sets the specified model as active by reloading the LLM service with the model path."""
    model_id = request.model_id
    logger.info(f"Received request to set active model: {model_id}")

    # Need model_manager functions here
    from app.scripts.model_manager import AVAILABLE_MODELS, MODELS_DIR, get_model_status

    if model_id not in AVAILABLE_MODELS:
        logger.warning(f"Set active request for unknown model_id: {model_id}")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    # Check if the model file exists before trying to set it active
    status_info = get_model_status(model_id)
    if status_info["status"] != "available":
        logger.warning(f"Cannot set model '{model_id}' active, status is: {status_info['status']}")
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not downloaded or failed. Status: {status_info['status']}")

    try:
        # Get the required model path
        model_info = AVAILABLE_MODELS[model_id]
        model_path_to_load = os.path.join(MODELS_DIR, model_info["filename"])

        # --- Reload the LLM service with the specific model path --- 
        try:
            logger.info(f"Attempting to reload LLMService with specific model path: {model_path_to_load}...")
            current_llm_service.reload_model(model_path=model_path_to_load)
            logger.info("LLMService reloaded successfully.")
            return {"message": f"Model '{model_id}' set as active and LLM service reloaded."}
        except Exception as reload_e:
            logger.error(f"Failed to reload LLM service for model {model_id} ({model_path_to_load}): {reload_e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to reload LLM service for model {model_id}: {reload_e}")
            
    except Exception as e:
        logger.error(f"Error setting active model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set active model {model_id}: {e}")

@router.get("/models/available")
async def list_available_models():
    """Lists all models defined in AVAILABLE_MODELS."""
    try:
        models = get_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/current")
async def get_current_model_route():
    """Gets the currently active model based on the .env configuration."""
    try:
        model_id, filename, config_path = check_current_model()
        if model_id:
            return {"model_id": model_id, "filename": filename, "config_path": config_path}
        else:
            raise HTTPException(status_code=404, detail="Current model configuration not found or invalid.")
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error checking current model: {str(e)}")

@router.get("/models/{model_id}/status")
async def get_model_status_route(model_id: str):
    """Checks the download status of a specific model."""
    try:
        status_info = get_model_status(model_id)
        if status_info["status"] == "not_found":
            raise HTTPException(status_code=404, detail=status_info["message"])
        return status_info
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")

# --- Chat Management Routes ---

@router.get("/config/llm",
             response_model=LlmConfigResponse,
             tags=["Admin & Config"],
             summary="Get current LLM configuration")
async def get_llm_configuration(
    current_user: dict = Depends(get_current_user) # Optional: Secure this endpoint
):
    """Returns the current LLM configuration settings."""
    try:
        # We can read directly from the settings object
        from app.core.config import get_settings # Local import OK here
        settings = get_settings()
        return LlmConfigResponse(
            backend=settings.LLM_BACKEND,
            model_id=settings.MODEL_ID,
            model_path=settings.MODEL_PATH,
            max_model_len=settings.MAX_MODEL_LEN,
            gpu_count=settings.GPU_COUNT
        )
    except Exception as e:
        logger.error(f"Error retrieving LLM configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve LLM configuration")

# Define the directory for saving refinement data
REFINEMENT_DATA_DIR = Path("app/data/improvement")
REFINEMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/chats/{chat_id}/messages/{message_index}/refine", status_code=status.HTTP_201_CREATED)
async def save_refined_message(
    chat_id: str,
    message_index: int,
    chat_service: ChatService = Depends(get_chat_service) # Or however ChatService is injected
):
    """
    Saves a specific user-assistant message pair from a chat for refinement/improvement.
    """
    logger.info(f"Attempting to save refinement data for chat {chat_id}, message index {message_index}")
    try:
        history = await chat_service.get_chat_history(chat_id)
        if not history or not history.get("messages"):
            logger.warning(f"Chat history not found or empty for chat_id: {chat_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat history not found or empty")

        messages = history["messages"]
        
        # message_index likely refers to the assistant's response.
        # We need the preceding user message and the assistant message at the index.
        # Check boundaries: index must be odd (assistant) and > 0, and within list bounds.
        if message_index <= 0 or message_index % 2 == 0 or message_index >= len(messages):
            logger.warning(f"Invalid message index {message_index} for chat {chat_id} with {len(messages)} messages.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid message index. Must be an odd number > 0 and less than the number of messages ({len(messages)}).",
            )

        user_message = messages[message_index - 1]
        assistant_message = messages[message_index]

        # Ensure the roles are correct (user followed by assistant)
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
             logger.warning(f"Unexpected message roles at index {message_index-1} ('{user_message.get('role')}') and {message_index} ('{assistant_message.get('role')}') in chat {chat_id}.")
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail="Message pair at index does not correspond to user -> assistant.",
             )

        refinement_data = {
            "chat_id": chat_id,
            "user_message": user_message.get("content"),
            "assistant_response": assistant_message.get("content"),
            "timestamp": datetime.now().isoformat(),
        }

        # Create a unique filename
        filename = f"refinement_{chat_id}_{message_index - 1}_{message_index}.json"
        filepath = REFINEMENT_DATA_DIR / filename

        try:
            with open(filepath, "w") as f:
                json.dump(refinement_data, f, indent=2)
            logger.info(f"Successfully saved refinement data to {filepath}")
            return {"message": "Refinement data saved successfully", "filename": filename}
        except IOError as e:
            logger.error(f"Failed to write refinement data to {filepath}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save refinement data")

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.exception(f"An unexpected error occurred while saving refinement data for chat {chat_id}, index {message_index}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

@router.post("/projects/{project_id}/chats",
             response_model=ChatSummaryResponse,
             tags=["Chats"],
             summary="Create a new chat within a project, checking access.",
             response_model_by_alias=False,
             status_code=status.HTTP_201_CREATED)
async def create_new_chat_for_project(
    project_id: str,
    chat_data: Optional[ChatCreateRequest] = None, # Optional body for chat name, etc.
    project_manager: ProjectManager = Depends(get_project_manager),
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(get_current_user)
):
    """Creates a new chat session associated with a specific project, checking access."""
    logger.info(f"Received request to create chat for project: {project_id} by user {current_user.get('id')}")
    try:
        # 1. Validate Project ID and Access
        # First check existence (get_project returns None if not found)
        project = project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Now check access using can_access_project
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden to create chat in project {project_id}")
             raise HTTPException(status_code=403, detail="You do not have permission to create chats in this project.")

        # 2. Create Chat using ChatService
        chat_name = chat_data.name if chat_data else None
        # Assume create_chat returns a dict with necessary info
        chat_info = chat_service.create_chat(project_id=project_id, chat_name=chat_name)

        if not chat_info or 'chat_id' not in chat_info:
             logger.error(f"Chat service failed to return valid info for project {project_id}")
             raise HTTPException(status_code=500, detail="Failed to get valid info for newly created chat.")

        new_chat_id = chat_info['chat_id']
        logger.info(f"Chat created successfully with id: {new_chat_id} for project {project_id}")

        # 3. Return response matching ChatSummaryResponse using the correct field names
        #    Ensure the keys match the Pydantic model definition
        return ChatSummaryResponse(
            id=chat_info.get('chat_id'),  # Map 'chat_id' from service to 'id' in response model
            project_id=chat_info.get('project_id', project_id),
            title=chat_info.get('name', f"Chat {new_chat_id[:8]}"), # Map 'name' from service to 'title'
            created_at=chat_info.get('created_at', datetime.now(timezone.utc).isoformat()),
            last_updated_at=chat_info.get('last_updated', datetime.now(timezone.utc).isoformat()) # Map 'last_updated' from service to 'last_updated_at'
        )

    except HTTPException as http_exc:
        raise http_exc # Re-raise known HTTP errors
    except KeyError as key_err:
         logger.error(f"Chat service create_chat response missing key: {key_err} for project {project_id}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Chat service response format error: missing '{key_err}'")
    except Exception as e:
        logger.error(f"Error creating chat for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {str(e)}")

@router.get("/projects/{project_id}/chats",
            response_model=List[ChatSummaryResponse], # Return a list of summaries
            tags=["Chats"],
            summary="List chats for a specific project, checking access.")
async def list_chats_for_project_route(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(get_current_user)
):
    """Retrieves a list of chat summaries associated with a given project, checking access."""
    logger.info(f"Request received to list chats for project: {project_id} by user {current_user.get('id')}")
    try:
        # 1. Validate Project ID and Access
        project = project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Project not found when listing chats: {project_id}")
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

        # Add access check
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden to list chats for project {project_id}")
             raise HTTPException(status_code=403, detail="You do not have permission to access chats in this project.")

        # 2. Fetch chats using ChatService
        chats_data = chat_service.list_chats_for_project(project_id) # Assuming this returns list of dicts
        logger.info(f"Found {len(chats_data)} chats for project {project_id}")

        # 3. Convert data to ChatSummaryResponse models
        chat_summaries = [
            ChatSummaryResponse(
                id=chat.get('chat_id'),
                project_id=chat.get('project_id'),
                title=chat.get('name', f"Chat {chat.get('chat_id')[:8]}" if chat.get('chat_id') else "Untitled Chat"), 
                created_at=chat.get('created_at'),
                last_updated_at=chat.get('last_updated') or chat.get('created_at')
            ) for chat in chats_data if chat.get('chat_id') 
        ]
        return chat_summaries

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error listing chats for project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list chats for project.")

@router.get("/chats/{chat_id}/history",
            response_model=ChatHistoryResponse,
            tags=["Chats"],
            summary="Get chat history, checking access")
async def get_chat_history_route(
    chat_id: str,
    limit: int = Query(100, ge=1, le=1000),
    chat_service: ChatService = Depends(get_chat_service),
    project_manager: ProjectManager = Depends(get_project_manager), # Need project manager for access check
    current_user: dict = Depends(get_current_user) # Secure endpoint
):
    """Retrieves the message history for a specific chat session, checking access."""
    logger.info(f"Request received for chat history: {chat_id} by user {current_user.get('id')}")
    try:
        # 1. Get chat summary to find its project ID
        # Assume get_chat_summary returns a dict like {'chat_id': ..., 'project_id': ...} or None
        chat_summary = await chat_service.get_chat_summary(chat_id)
        if not chat_summary or not chat_summary.get('project_id'):
            logger.warning(f"Chat {chat_id} not found or has no associated project_id.")
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found or project association missing.")
            
        project_id = chat_summary['project_id']
        
        # 2. Check if user can access the associated project
        if not project_manager.can_access_project(project_id, current_user["id"]):
             logger.warning(f"User {current_user.get('id')} forbidden to access history for chat {chat_id} in project {project_id}")
             raise HTTPException(status_code=403, detail="You do not have permission to access this chat history.")

        # 3. Get actual history (now returns just the list of messages)
        history_messages = chat_service.get_chat_history(chat_id, limit=limit) 
        # Note: The new service method returns [] if chat doesn't exist or fails, 
        # so the check below might not be strictly necessary but is harmless.
        # if not history_messages:
        #     logger.error(f"Chat history inconsistency for {chat_id}. Summary found but history failed.")
        #     raise HTTPException(status_code=404, detail=f"Chat history for {chat_id} not found.")

        # 4. Return response using the history list directly
        return ChatHistoryResponse(
            chat_id=chat_id, # Use the validated chat_id
            messages=history_messages,
            project_id=project_id # Include the verified project_id
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error retrieving chat history for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")

# --- NEW: Delete Chat Endpoint ---
@router.delete("/projects/{project_id}/chats/{chat_id}", # Modified route
               status_code=status.HTTP_204_NO_CONTENT,
               tags=["Chats"],
               summary="Delete a chat session")
async def delete_chat_route(
    project_id: str, # Added project_id from path
    chat_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user)
):
    """Deletes a chat session and removes its association from its project."""
    logger.info(f"Request received to delete chat: {chat_id} from project {project_id} by user {current_user.get('id')}")
    try:
        # 1. Check Access Control (User must own the project)
        project = project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Attempt to delete chat {chat_id} from non-existent project {project_id}.")
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")
        elif project.get("owner_id") != current_user.get("id"):
            logger.warning(f"User {current_user.get('id')} attempted to delete chat {chat_id} from project {project_id} owned by {project.get('owner_id')}")
            raise HTTPException(status_code=403, detail="You do not have permission to delete chats in this project.")

        # 2. Check if chat actually belongs to the project (optional, but good practice)
        # This might require ChatService or ProjectManager to verify association.
        # For now, we assume if the user owns the project, they can delete any chat ID provided under it.
        # A more robust check could be added here if needed.
        
        # 3. Delete the chat using ChatService
        success = await chat_service.delete_chat(chat_id)

        if not success:
            # Improved check: Ask ChatService if chat exists before declaring failure
            chat_exists = await chat_service.get_chat_summary(chat_id) is not None
            if chat_exists:
                 logger.error(f"Chat service failed to delete existing chat {chat_id} from project {project_id}.")
                 raise HTTPException(status_code=500, detail=f"Failed to delete chat {chat_id}.")
            else:
                 logger.warning(f"Chat service reported failure deleting chat {chat_id} from project {project_id}, but it was already gone.")
                 # Treat as success if chat is gone
                 pass

        logger.info(f"Successfully deleted chat {chat_id} from project {project_id} (or it was already gone).")

        # 4. Remove chat ID from project's list (if project still exists)
        try:
            project_still_exists = project_manager.get_project(project_id) is not None
            if project_still_exists:
                 project_manager.remove_conversation_from_project(project_id, chat_id)
                 logger.info(f"Removed chat ID {chat_id} from project {project_id}'s list.")
            else:
                 logger.warning(f"Project {project_id} disappeared before chat ID {chat_id} could be removed from its list.")
        except Exception as remove_err:
            logger.error(f"Failed to remove chat ID {chat_id} from project {project_id} list after deletion: {remove_err}", exc_info=True)
            # Continue anyway, as chat is deleted.

        return None # Return None for 204 No Content

    except HTTPException as http_exc:
        raise http_exc
    # Removed specific AttributeError check for find_project_by_chat_id
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id} from project {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete chat session.")
