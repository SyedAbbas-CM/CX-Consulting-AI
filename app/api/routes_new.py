# app/api/routes.py
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.api.auth import get_current_user
from app.api.dependencies import get_memory_manager, get_project_manager, get_rag_engine
from app.api.middleware.admin import admin_required
from app.api.models import (
    ConversationInfo,
    ConversationsResponse,
    CXStrategyRequest,
    DeliverableResponse,
    DocumentUploadResponse,
    JourneyMapRequest,
    Project,
    ProjectCreateRequest,
    ProjectDocument,
    ProjectDocumentsResponse,
    ProjectsResponse,
    QuestionRequest,
    QuestionResponse,
    ROIAnalysisRequest,
    SearchResult,
)
from app.services.memory_manager import MemoryManager
from app.services.project_manager import ProjectManager
from app.services.rag_engine import RagEngine

# Configure logger
logger = logging.getLogger("cx_consulting_ai.api.routes")

# Create router
router = APIRouter(prefix="/api")


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    is_global: bool = Form(False),
    rag_engine: RagEngine = Depends(get_rag_engine),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Upload a document to the knowledge base."""
    try:
        # Check if either project_id is provided or document is global
        if not project_id and not is_global:
            raise HTTPException(
                status_code=400,
                detail="Either project_id must be provided or document must be marked as global",
            )

        # If project_id is provided, validate it exists
        if project_id:
            project = project_manager.get_project(project_id)
            if not project:
                raise HTTPException(
                    status_code=404, detail=f"Project {project_id} not found"
                )

        # Check file size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        max_size_mb = 50  # 50 MB max

        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.2f} MB. Maximum size: {max_size_mb} MB",
            )

        # Process document
        result = await rag_engine.process_document(content, file.filename)

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["error"])

        # If project_id is provided, add document to project
        if project_id:
            project_manager.add_document_to_project(project_id, result["document_id"])

        return DocumentUploadResponse(
            filename=file.filename,
            document_id=result["document_id"],
            project_id=project_id,
            is_global=is_global,
            chunks_created=result["chunks_created"],
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Ask a question to the CX Consulting AI."""
    try:
        start_time = time.time()

        # Get or create conversation
        conversation_id = request.conversation_id
        project_id = request.project_id

        if not conversation_id:
            # Create new conversation
            conversation_id = memory_manager.create_conversation(project_id)
        elif not memory_manager.get_conversation(conversation_id):
            # If conversation ID is provided but doesn't exist, create it
            conversation_id = memory_manager.create_conversation(project_id)
        elif (
            project_id
            and memory_manager.get_conversation_project(conversation_id) != project_id
        ):
            # If project ID is provided and different from current association, update it
            memory_manager.set_conversation_project(conversation_id, project_id)

            # Also update the project's conversation list
            if project_manager.get_project(project_id):
                project_manager.add_conversation_to_project(project_id, conversation_id)

        # Get conversation history
        conversation_history = memory_manager.get_formatted_history(conversation_id)

        # Add user message to memory
        memory_manager.add_message(
            conversation_id=conversation_id, role="user", content=request.query
        )

        # Generate answer
        response = await rag_engine.ask(
            question=request.query, conversation_id=conversation_id
        )

        # Add assistant message to memory
        memory_manager.add_message(
            conversation_id=conversation_id, role="assistant", content=response
        )

        # Prepare sources for response (placeholder for now)
        sources = []

        processing_time = time.time() - start_time

        return QuestionResponse(
            answer=response,
            conversation_id=conversation_id,
            project_id=memory_manager.get_conversation_project(conversation_id),
            sources=sources,
            processing_time=processing_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )


@router.post("/cx-strategy", response_model=DeliverableResponse)
async def generate_cx_strategy(
    request: CXStrategyRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Generate a CX strategy document."""
    try:
        start_time = time.time()

        # Get or create conversation
        conversation_id = request.conversation_id
        project_id = request.project_id

        if not conversation_id:
            conversation_id = memory_manager.create_conversation(project_id)
        elif not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(project_id)
        elif (
            project_id
            and memory_manager.get_conversation_project(conversation_id) != project_id
        ):
            # Update project association
            memory_manager.set_conversation_project(conversation_id, project_id)

            # Update project's conversation list
            if project_manager.get_project(project_id):
                project_manager.add_conversation_to_project(project_id, conversation_id)

        # Generate strategy
        response, conversation_id, tokens = await rag_engine.create_proposal(
            client_info=f"{request.client_name} ({request.industry})",
            requirements=request.challenges,
            conversation_id=conversation_id,
        )

        # Store document if project_id is provided
        document_id = None
        if project_id:
            try:
                # Create a document in the project
                document_id = project_manager.create_document(
                    project_id=project_id,
                    title=f"CX Strategy for {request.client_name}",
                    content=response,
                    document_type="cx_strategy",
                    metadata={
                        "client_name": request.client_name,
                        "industry": request.industry,
                        "challenges": request.challenges,
                        "conversation_id": conversation_id,
                        "tokens": tokens,
                    },
                )
            except Exception as doc_error:
                logger.error(f"Error storing document: {str(doc_error)}")
                # Continue without failing the request

        processing_time = time.time() - start_time

        return DeliverableResponse(
            content=response,
            conversation_id=conversation_id,
            project_id=project_id,
            document_id=document_id,
            processing_time=processing_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating CX strategy: {str(e)}"
        )


@router.post("/roi-analysis", response_model=DeliverableResponse)
async def generate_roi_analysis(
    request: ROIAnalysisRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Generate an ROI analysis."""
    try:
        start_time = time.time()

        # Get or create conversation
        conversation_id = request.conversation_id
        project_id = request.project_id

        if not conversation_id:
            conversation_id = memory_manager.create_conversation(project_id)
        elif not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(project_id)
        elif (
            project_id
            and memory_manager.get_conversation_project(conversation_id) != project_id
        ):
            # Update project association
            memory_manager.set_conversation_project(conversation_id, project_id)

            # Update project's conversation list
            if project_manager.get_project(project_id):
                project_manager.add_conversation_to_project(project_id, conversation_id)

        # Generate ROI analysis
        response, conversation_id, tokens = await rag_engine.create_roi_analysis(
            client_info=f"{request.client_name} ({request.industry})",
            project_details=f"{request.project_description}\n\nCurrent metrics: {request.current_metrics}",
            conversation_id=conversation_id,
        )

        # Store document if project_id is provided
        document_id = None
        if project_id:
            try:
                # Create a document in the project
                document_id = project_manager.create_document(
                    project_id=project_id,
                    title=f"ROI Analysis for {request.client_name}",
                    content=response,
                    document_type="roi_analysis",
                    metadata={
                        "client_name": request.client_name,
                        "industry": request.industry,
                        "project_description": request.project_description,
                        "current_metrics": request.current_metrics,
                        "conversation_id": conversation_id,
                        "tokens": tokens,
                    },
                )
            except Exception as doc_error:
                logger.error(f"Error storing document: {str(doc_error)}")
                # Continue without failing the request

        processing_time = time.time() - start_time

        return DeliverableResponse(
            content=response,
            conversation_id=conversation_id,
            project_id=project_id,
            document_id=document_id,
            processing_time=processing_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating ROI analysis: {str(e)}"
        )


@router.post("/journey-map", response_model=DeliverableResponse)
async def generate_journey_map(
    request: JourneyMapRequest,
    rag_engine: RagEngine = Depends(get_rag_engine),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Generate a customer journey map."""
    try:
        start_time = time.time()

        # Get or create conversation
        conversation_id = request.conversation_id
        project_id = request.project_id

        if not conversation_id:
            conversation_id = memory_manager.create_conversation(project_id)
        elif not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(project_id)
        elif (
            project_id
            and memory_manager.get_conversation_project(conversation_id) != project_id
        ):
            # Update project association
            memory_manager.set_conversation_project(conversation_id, project_id)

            # Update project's conversation list
            if project_manager.get_project(project_id):
                project_manager.add_conversation_to_project(project_id, conversation_id)

        # Generate journey map
        response, conversation_id, tokens = await rag_engine.create_journey_map(
            client_info=f"{request.client_name} ({request.industry})",
            journey_type=request.journey_type,
            touchpoints=request.touchpoints,
            conversation_id=conversation_id,
        )

        # Store document if project_id is provided
        document_id = None
        if project_id:
            try:
                # Create a document in the project
                document_id = project_manager.create_document(
                    project_id=project_id,
                    title=f"Journey Map for {request.client_name}",
                    content=response,
                    document_type="journey_map",
                    metadata={
                        "client_name": request.client_name,
                        "industry": request.industry,
                        "journey_type": request.journey_type,
                        "touchpoints": request.touchpoints,
                        "conversation_id": conversation_id,
                        "tokens": tokens,
                    },
                )
            except Exception as doc_error:
                logger.error(f"Error storing document: {str(doc_error)}")
                # Continue without failing the request

        processing_time = time.time() - start_time

        return DeliverableResponse(
            content=response,
            conversation_id=conversation_id,
            project_id=project_id,
            document_id=document_id,
            processing_time=processing_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating journey map: {str(e)}"
        )


@router.get("/conversations", response_model=ConversationsResponse)
async def list_conversations(
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: dict = Depends(get_current_user),
):
    """List all conversations."""
    try:
        conversations = memory_manager.list_conversations()
        return ConversationsResponse(conversations=conversations)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationInfo)
async def get_conversation(
    conversation_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: dict = Depends(get_current_user),
):
    """Get a specific conversation."""
    try:
        conversation = memory_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id} not found"
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving conversation: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: dict = Depends(get_current_user),
):
    """Delete a specific conversation."""
    try:
        if not memory_manager.get_conversation(conversation_id):
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id} not found"
            )

        memory_manager.delete_conversation(conversation_id)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Conversation {conversation_id} deleted",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting conversation: {str(e)}"
        )


@router.get("/projects", response_model=ProjectsResponse)
async def list_projects(
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """List all projects."""
    try:
        projects = project_manager.list_projects()
        return ProjectsResponse(projects=projects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")


@router.post("/projects", response_model=Project)
async def create_project(
    request: ProjectCreateRequest,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Create a new project."""
    try:
        project = project_manager.create_project(
            name=request.name,
            description=request.description,
            client_name=request.client_name,
            industry=request.industry,
        )
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating project: {str(e)}")


@router.get("/projects/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Get a specific project."""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )
        return project
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving project: {str(e)}"
        )


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """Delete a specific project."""
    try:
        if not project_manager.get_project(project_id):
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )

        project_manager.delete_project(project_id)
        return JSONResponse(
            content={"status": "success", "message": f"Project {project_id} deleted"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")


@router.get("/projects/{project_id}/documents", response_model=ProjectDocumentsResponse)
async def list_project_documents(
    project_id: str,
    project_manager: ProjectManager = Depends(get_project_manager),
    current_user: dict = Depends(get_current_user),
):
    """List all documents in a project."""
    try:
        if not project_manager.get_project(project_id):
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )

        documents = project_manager.list_project_documents(project_id)
        return ProjectDocumentsResponse(documents=documents)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing project documents: {str(e)}"
        )


@router.get("/admin/system-info")
@admin_required
async def get_system_info(current_user: dict = Depends(get_current_user)):
    """Get system information (admin only)."""
    try:
        # Collect system information
        system_info = {
            "app_version": os.environ.get("APP_VERSION", "1.0.0"),
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "python_version": os.environ.get("PYTHON_VERSION"),
            "uptime": time.time() - float(os.environ.get("START_TIME", time.time())),
            "timestamp": datetime.now().isoformat(),
        }

        return JSONResponse(content=system_info)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving system information: {str(e)}"
        )
