from typing import Optional
from pydantic import BaseModel

class DocumentUploadResponse(BaseModel):
    filename: str
    document_id: str
    chunks_created: int
    project_id: Optional[str] = None
    is_global: bool = False
    message: Optional[str] = None 