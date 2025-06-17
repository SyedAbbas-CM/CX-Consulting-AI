import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_chat_service, get_document_service
from app.services.project_handler import ProjectHandler

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}},
)


# Dependency to get project handler leveraging singleton services on app.state
# We resolve ChatService and DocumentService via existing dependency providers.
def get_project_handler(
    chat_service=Depends(get_chat_service),
    document_service=Depends(get_document_service),
):
    return ProjectHandler(chat_service=chat_service, document_service=document_service)


# Models
class ProjectCreate(BaseModel):
    name: str = Field(..., description="Project name")
    description: str = Field("", description="Project description")
    metadata: Optional[dict] = Field(None, description="Additional project metadata")


class ConversationLink(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID to link to project")


class DocumentLink(BaseModel):
    document_id: str = Field(..., description="Document ID to link to project")
    metadata: Optional[dict] = Field(None, description="Additional document metadata")


class Project(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    metadata: dict
    conversations: List[str]
    documents: List[dict]


class ProjectSummary(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    conversation_count: int
    conversations: List[dict]
    document_count: int
    documents: List[dict]


# Routes
@router.post("/", response_model=Project)
async def create_project(
    project_data: ProjectCreate,
    project_handler: ProjectHandler = Depends(get_project_handler),
):
    """Create a new project"""
    project = project_handler.create_project(
        name=project_data.name,
        description=project_data.description,
        metadata=project_data.metadata,
    )
    return project


@router.get("/", response_model=List[Project])
async def list_projects(project_handler: ProjectHandler = Depends(get_project_handler)):
    """List all projects"""
    return project_handler.list_projects()


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str, project_handler: ProjectHandler = Depends(get_project_handler)
):
    """Get project details by ID"""
    project = project_handler.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/{project_id}/summary", response_model=ProjectSummary)
async def get_project_summary(
    project_id: str, project_handler: ProjectHandler = Depends(get_project_handler)
):
    """Get detailed project summary including conversations and documents"""
    summary = project_handler.get_project_summary(project_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Project not found")
    return summary


@router.post("/{project_id}/conversations", response_model=dict)
async def add_conversation(
    project_id: str,
    link_data: ConversationLink,
    project_handler: ProjectHandler = Depends(get_project_handler),
):
    """Link a conversation to a project"""
    success = project_handler.add_conversation_to_project(
        project_id=project_id, conversation_id=link_data.conversation_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "success", "message": "Conversation linked to project"}


@router.get("/{project_id}/conversations", response_model=List[str])
async def get_project_conversations(
    project_id: str, project_handler: ProjectHandler = Depends(get_project_handler)
):
    """Get all conversation IDs for a project"""
    project = project_handler.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_handler.get_project_conversations(project_id)


@router.post("/{project_id}/documents", response_model=dict)
async def add_document(
    project_id: str,
    link_data: DocumentLink,
    project_handler: ProjectHandler = Depends(get_project_handler),
):
    """Link a document to a project"""
    success = project_handler.add_document_to_project(
        project_id=project_id,
        document_id=link_data.document_id,
        metadata=link_data.metadata,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "success", "message": "Document linked to project"}
