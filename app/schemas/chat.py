from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatCreateRequest(BaseModel):
    name: Optional[str] = None  # Optional chat name
    journey_type: Optional[str] = Field(
        None,
        description="Optional journey type (e.g., roi_analysis, interview_prep, journey_mapping)",
        pattern="^[a-zA-Z0-9_\-]+$",
    )


class ChatSummaryResponse(BaseModel):
    id: str = Field(..., alias="chat_id")  # Frontend expects 'id', maps from 'chat_id'
    project_id: str
    title: Optional[str] = Field(
        None, alias="name"
    )  # Frontend expects 'title', maps from 'name'
    created_at: str
    last_updated_at: str = Field(
        ..., alias="last_updated"
    )  # Frontend expects 'last_updated_at', maps from 'last_updated'

    class Config:
        populate_by_name = True  # Allow using alias names for input


class ChatCreateResponse(ChatSummaryResponse):
    # Currently same as summary, might add more later
    pass


class RefinementResponse(BaseModel):
    message: str
    interaction_id: str
    file_path: str


# Model for returning chat history (used in routes.py)
class ChatHistoryResponse(BaseModel):
    messages: List[Dict[str, Any]]


# --- Add Missing Chat Message Schemas ---
class ChatMessageCreateRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")  # Enforce user or assistant
    content: str


class ChatMessageResponse(BaseModel):
    role: str
    content: str
    timestamp: str  # ISO format string


# --- End Added Schemas ---
