# app/api/models.py
import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# --- Added for Workstream 1: Document Generation Config ---
class DocumentGenerationConfig(BaseModel):
    """Configuration for generating a specific deliverable document."""

    deliverable_type: Literal[
        "proposal", "roi_analysis", "journey_map", "cx_strategy"
    ] = Field(..., description="The type of deliverable to generate.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters required for the specified deliverable type, e.g., {'client_name': 'Acme Corp', 'industry': 'Tech'}.",
    )


# --- End Workstream 1 ---


class UserCreate(BaseModel):
    """Request model for user creation."""

    username: str = Field(..., description="Username", min_length=3, max_length=50)
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    full_name: Optional[str] = Field(None, description="Full name")
    company: Optional[str] = Field(None, description="Company name")


class UserLogin(BaseModel):
    """Request model for user login."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class User(BaseModel):
    """Model for user information."""

    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    company: Optional[str] = Field(None, description="Company name")
    is_active: bool = Field(True, description="Whether the user is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class Token(BaseModel):
    """Model for authentication tokens."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_at: str = Field(..., description="Token expiration timestamp")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")


class UserUpdateRequest(BaseModel):
    """Request model for updating user information."""

    email: Optional[EmailStr] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    company: Optional[str] = Field(None, description="Company name")
    password: Optional[str] = Field(None, description="New password", min_length=8)


class UsersResponse(BaseModel):
    """Response model for listing users."""

    users: List[User] = Field(..., description="List of users")
    count: int = Field(..., description="Total number of users")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""

    project_id: Optional[str] = Field(
        None, description="Project ID this document belongs to"
    )
    is_global: bool = Field(
        False, description="Whether the document is globally accessible"
    )
    document_type: Optional[str] = Field(
        None, description="Type of document (pdf, txt, docx, etc.)"
    )
    description: Optional[str] = Field(
        None, description="Brief description of the document"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Tags for categorizing the document"
    )


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    filename: str = Field(..., description="Name of the uploaded file")
    document_id: str = Field(..., description="Document ID for retrieval")
    chunks_created: int = Field(
        ..., description="Number of chunks created from the document"
    )
    project_id: Optional[str] = Field(
        None, description="Project ID this document belongs to"
    )
    is_global: bool = Field(
        False, description="Whether the document is globally accessible"
    )
    message: Optional[str] = Field(None, description="Status message")


class QuestionRequest(BaseModel):
    """Request model for asking a question or generating a document."""

    query: str = Field(
        ..., description="The user's question or main input for document generation."
    )
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for memory"
    )
    project_id: Optional[str] = Field(
        None, description="Project ID for grouping conversations"
    )
    # Removed include_global as per later RAG logic, will be handled by retriever agent
    # include_global: bool = Field(False, description="Whether to include global documents in RAG search")

    # --- Added for Workstream 1 / 5 ---
    mode: Literal["chat", "document"] = Field(
        "chat",
        description="Mode of operation: 'chat' for QA, 'document' for deliverable generation.",
    )
    doc_config: Optional[DocumentGenerationConfig] = Field(
        None,
        description="Configuration for document generation, required if mode is 'document'.",
    )
    active_deliverable_type: Optional[str] = Field(
        None,
        description="When set, instructs backend to generate/fill that deliverable via conversation context.",
    )
    # --- End Workstream 1 / 5 ---


class OriginType(str, Enum):
    """Enum for the origin of a search result document."""

    PROJECT_UPLOAD = "project_upload"
    GLOBAL_KB = "global_kb"
    TEMPLATE = "template"
    UNKNOWN = "unknown"


class SearchResult(BaseModel):
    """Model for search result information."""

    source: str = Field(
        ..., description="Source document identifier (e.g., document ID or name)"
    )
    score: float = Field(..., description="Relevance score")
    text_snippet: str = Field(..., description="Text snippet from the document")
    origin_type: OriginType = Field(
        OriginType.UNKNOWN,
        description="Indicates the origin of the source document (e.g., uploaded to project, global KB).",
    )


class QuestionResponse(BaseModel):
    """Response model for question answering."""

    answer: str = Field(..., description="The generated answer")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for memory"
    )
    project_id: Optional[str] = Field(
        None, description="Project ID for grouping conversations"
    )
    sources: List[SearchResult] = Field([], description="Source documents used")
    processing_time: float = Field(..., description="Processing time in seconds")


class CXStrategyRequest(BaseModel):
    """Request model for CX strategy generation."""

    client_name: str = Field(..., description="Client name")
    industry: str = Field(..., description="Client industry")
    challenges: str = Field(..., description="Current CX challenges")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for memory"
    )
    project_id: Optional[str] = Field(
        None, description="Project ID for grouping conversations"
    )


class ROIAnalysisRequest(BaseModel):
    """Request model for ROI analysis generation."""

    client_name: str = Field(..., description="Client name")
    industry: str = Field(..., description="Client industry")
    project_description: str = Field(..., description="Project description")
    current_metrics: str = Field(..., description="Current metrics")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for memory"
    )
    project_id: Optional[str] = Field(
        None, description="Project ID for grouping conversations"
    )


class JourneyMapRequest(BaseModel):
    """Request model for journey map generation."""

    client_name: str = Field(..., description="Client name")
    industry: str = Field(..., description="Client industry")
    persona: str = Field(..., description="Customer persona")
    scenario: str = Field(..., description="Journey scenario")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for memory"
    )
    project_id: Optional[str] = Field(
        None, description="Project ID for grouping conversations"
    )


class DeliverableResponse(BaseModel):
    """Response model for deliverable generation."""

    content: str = Field(..., description="The generated deliverable content")
    conversation_id: str = Field(..., description="Conversation ID for memory")
    project_id: Optional[str] = Field(
        None, description="Project ID for the deliverable"
    )
    document_id: Optional[str] = Field(None, description="Document ID for retrieval")
    processing_time: float = Field(..., description="Processing time in seconds")


class ConversationInfo(BaseModel):
    """Model for conversation information."""

    id: str = Field(..., description="Conversation ID")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages")
    project_id: Optional[str] = Field(
        None, description="Project ID this conversation belongs to"
    )


class ConversationsResponse(BaseModel):
    """Response model for listing conversations."""

    conversations: List[ConversationInfo] = Field(
        ..., description="List of conversations"
    )
    count: int = Field(..., description="Total number of conversations")


class ProjectDocument(BaseModel):
    """Model for project document information."""

    id: str = Field(..., description="Document ID")
    project_id: str = Field(..., description="Project ID this document belongs to")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    document_type: str = Field(
        ..., description="Type of document (strategy, roi, journey_map, etc.)"
    )
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Project(BaseModel):
    """Model for project information."""

    id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    client_name: Optional[str] = Field(None, description="Client name")
    industry: Optional[str] = Field(None, description="Client industry")
    description: Optional[str] = Field(None, description="Project description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    owner_id: str = Field(..., description="User ID of the project owner")
    shared_with: List[str] = Field(
        default_factory=list, description="User IDs this project is shared with"
    )
    conversation_ids: List[str] = Field(
        default_factory=list, description="Conversation IDs in this project"
    )
    document_ids: List[str] = Field(
        default_factory=list, description="Document IDs in this project"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ProjectCreateRequest(BaseModel):
    """Request model for creating a project."""

    name: str = Field(..., description="Project name")
    client_name: Optional[str] = Field(None, description="Client name")
    industry: Optional[str] = Field(None, description="Client industry")
    description: Optional[str] = Field(None, description="Project description")
    shared_with: Optional[List[str]] = Field(
        None, description="User IDs to share the project with"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ProjectsResponse(BaseModel):
    """Response model for listing projects."""

    projects: List[Project] = Field(..., description="List of projects")
    count: int = Field(..., description="Total number of projects")


class ProjectDocumentsResponse(BaseModel):
    """Response model for listing project documents."""

    documents: List[ProjectDocument] = Field(
        ..., description="List of project documents"
    )
    count: int = Field(..., description="Total number of documents")


class RefineRequest(BaseModel):
    """Request model for refining a document."""

    prompt: str = Field(..., description="User prompt/instructions for refinement.")
    replace_embeddings: bool = Field(
        False,
        description="Whether to replace existing embeddings (delete then add) or just update.",
    )


class ShareProjectRequest(BaseModel):
    """Request model for sharing a project with other users."""

    user_ids: List[str] = Field(..., description="User IDs to share the project with")


# --- New Deliverable Generation Request Model ---
class GenerateDeliverableRequest(BaseModel):
    """Request model for triggering deliverable generation."""

    project_id: str = Field(..., description="The project context for the deliverable.")
    deliverable_type: str = Field(
        ...,
        description="Type of deliverable to generate (e.g., roi_deck, cx_strategy_guide).",
    )
    user_input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs of user input to fill template placeholders.",
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation ID to associate with this generation."
    )


class JobQueuedResponse(BaseModel):
    job_id: str
    message: str = "File queued for background processing."
