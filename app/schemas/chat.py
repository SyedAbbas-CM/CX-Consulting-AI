from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatCreateRequest(BaseModel):
    name: Optional[str] = None # Optional chat name

class ChatSummaryResponse(BaseModel):
    id: str = Field(..., alias="chat_id") # Frontend expects 'id', maps from 'chat_id'
    project_id: str
    title: Optional[str] = Field(None, alias="name") # Frontend expects 'title', maps from 'name'
    created_at: str
    last_updated_at: str = Field(..., alias="last_updated") # Frontend expects 'last_updated_at', maps from 'last_updated'

    class Config:
        populate_by_name = True # Allow using alias names for input

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