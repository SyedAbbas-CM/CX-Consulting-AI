from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from app.api.models import DeliverableResponse
from app.services.rag_engine import RagEngine
from app.templates.deliverable_templates import DeliverableTemplates
from app.api.auth import get_current_user
import logging
import uuid
import time
import json

# Configure logger
logger = logging.getLogger("cx_consulting_ai.api.deliverables")

# Create router
router = APIRouter(tags=["Deliverables"])

# Initialize templates
deliverable_templates = DeliverableTemplates()

@router.get("/template-types", response_model=List[str])
async def get_template_types(current_user: dict = Depends(get_current_user)):
    """
    Get a list of all available deliverable template types.
    
    Args:
        current_user: Current user data
        
    Returns:
        List of template types
    """
    return deliverable_templates.get_template_list()

@router.get("/template/{template_type}", response_model=str)
async def get_template(
    template_type: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get a specific deliverable template.
    
    Args:
        template_type: Type of template to retrieve
        current_user: Current user data
        
    Returns:
        Template string
    """
    template = deliverable_templates.get_template(template_type)
    if template == "Template not found.":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template type '{template_type}' not found"
        )
    
    return template

@router.post("/generate/{deliverable_type}", response_model=DeliverableResponse)
async def generate_deliverable(
    deliverable_type: str,
    parameters: Dict[str, Any],
    conversation_id: Optional[str] = None,
    project_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    rag_engine: RagEngine = Depends(lambda: get_rag_engine())
):
    """
    Generate a deliverable using the specified template.
    
    Args:
        deliverable_type: Type of deliverable to generate
        parameters: Parameters to use for generation
        conversation_id: Optional conversation ID
        project_id: Optional project ID
        current_user: Current user data
        rag_engine: RAG engine instance
        
    Returns:
        Generated deliverable
    """
    start_time = time.time()
    
    # Check if deliverable type exists
    template = deliverable_templates.get_template(deliverable_type)
    if template == "Template not found.":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deliverable type '{deliverable_type}' not found"
        )
    
    # Log generation request
    logger.info(f"Generating {deliverable_type} deliverable for user {current_user['username']}")
    
    try:
        # Create or get conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Prepare prompt
        query = f"Generate a {deliverable_type} deliverable with the following parameters: {json.dumps(parameters)}"
        
        # Generate content
        content = await rag_engine.ask(query, conversation_id, project_id)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create deliverable ID
        document_id = str(uuid.uuid4())
        
        # Store the deliverable (we'll implement this in the "Data Storage for Generated Outputs" section)
        # store_deliverable(document_id, deliverable_type, content, parameters, current_user["id"], project_id)
        
        # Return response
        return {
            "content": content,
            "conversation_id": conversation_id,
            "project_id": project_id,
            "document_id": document_id,
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error generating deliverable: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating deliverable: {str(e)}"
        )

def get_rag_engine() -> RagEngine:
    """Get RAG engine from app state."""
    from app.main import app
    return app.state.rag_engine 