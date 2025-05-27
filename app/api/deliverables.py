import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename

from app.api.auth import get_current_user
from app.api.dependencies import (
    CurrentUserDep,
    DeliverableServiceDep,
    DocumentServiceDep,
    ProjectManagerDep,
    TemplateServiceDep,
)
from app.api.models import DeliverableResponse, GenerateDeliverableRequest
from app.core.config import get_settings
from app.services.deliverable_service import DeliverableService
from app.services.project_manager import ProjectManager
from app.services.template_service import TemplateService

# Configure logger
logger = logging.getLogger("cx_consulting_ai.api.deliverables")

# Create router
router = APIRouter(tags=["Deliverables"])


@router.get("/template-types", response_model=List[str])
async def get_template_types(
    template_service: TemplateServiceDep, current_user: CurrentUserDep
):
    """
    Get a list of all available deliverable template types based on files
    in the configured templates directory (e.g., .md files).
    """
    try:
        # Use TemplateService to list templates based on .md extension
        templates_dict = template_service.list_available_templates(
            file_extensions=[".md"]
        )
        template_names = list(templates_dict.keys())
        logger.info(f"Returning available deliverable template types: {template_names}")
        return template_names
    except Exception as e:
        logger.error(f"Error listing template types: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve template types",
        )


@router.get("/template/{template_type}", response_model=str)
async def get_template(
    template_type: str,
    template_service: TemplateServiceDep,
    current_user: CurrentUserDep,
):
    """
    Get the raw content of a specific deliverable template file (.md).
    """
    try:
        # Construct the expected filename
        template_filename = f"{template_type}.md"
        template_path = template_service.templates_dir / template_filename

        if not template_path.is_file():
            logger.warning(f"Template file not found: {template_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template type '{template_type}' (.md) not found",
            )

        # Read the raw content
        content = template_path.read_text(encoding="utf-8")
        logger.info(f"Returning content for template: {template_filename}")
        return content

    except HTTPException:
        raise  # Re-raise specific HTTP errors
    except Exception as e:
        logger.error(
            f"Error reading template file {template_type}.md: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read template file '{template_type}.md'",
        )


@router.post("/projects/{project_id}/upload_pdf", status_code=status.HTTP_201_CREATED)
async def upload_project_pdf(
    project_id: str,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
    document_service: DocumentServiceDep,
    file: UploadFile = File(...),
):
    """
    Uploads a PDF file to a specific project.
    The file will be stored in the project's dedicated folder.
    """
    settings = get_settings()
    user_id = current_user["id"]
    logger.info(
        f"Received request to upload PDF '{file.filename}' to project '{project_id}' by user '{user_id}'."
    )

    # 1. Check Project Access
    if not project_manager.can_access_project(project_id, user_id):
        logger.warning(
            f"User {user_id} forbidden to upload PDF to project {project_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this project.",
        )

    # 2. Validate file type (optional but good practice)
    if file.content_type != "application/pdf":
        logger.warning(
            f"Invalid file type uploaded: {file.content_type}. Expected PDF."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed.",
        )

    # 3. Define storage path
    # Ensure base project directory exists (ProjectManager might do this, but good to be safe)
    base_project_path = Path(settings.PROJECT_DIR) / project_id
    upload_dir = base_project_path / "uploaded_files"

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Could not create upload directory {upload_dir}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create upload directory on server.",
        )

    # Sanitize filename (basic sanitization)
    safe_filename = secure_filename(file.filename)
    file_path = upload_dir / safe_filename

    # Prevent overwriting for simplicity, or implement versioning/unique naming
    if file_path.exists():
        # Add a timestamp or UUID to make filename unique
        stem = file_path.stem
        suffix = file_path.suffix
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = f"{stem}_{unique_id}{suffix}"
        file_path = upload_dir / safe_filename

    # 4. Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(
            f"PDF '{safe_filename}' uploaded successfully to '{file_path}' for project '{project_id}'."
        )

        # 5. Process and ingest the PDF (synchronous for now)
        # Note: If DeliverableServiceDep is not actually DocumentService, this needs to be DocumentServiceDep
        # Assuming for now that the type hint DeliverableServiceDep can resolve to DocumentService or
        # that we will adjust this dependency injection later.
        # For the purpose of this step, we assume document_service is an instance of DocumentService.

        # Correct type hint for document_service is needed. For now, we assume it works or will be fixed.
        # from app.api.dependencies import DocumentServiceDep (this would be ideal)

        ingestion_metadata = {
            "uploaded_by_user_id": user_id,
            "original_filename": file.filename,
        }

        # Construct the full path for the service method using settings
        absolute_file_path = (
            Path(settings.PROJECT_DIR) / project_id / "uploaded_files" / safe_filename
        )

        # We need to ensure document_service is actually DocumentService.
        # This might require changing the dependency or adding a new one.
        # Let's assume we have a DocumentService instance. If not, this is a placeholder for that logic.
        try:
            from app.services.document_service import (  # Temporary direct import for clarity
                DocumentService,
            )

            if isinstance(document_service, DocumentService):  # Check instance type
                success = document_service.process_and_ingest_pdf_from_path(
                    file_path=absolute_file_path,
                    project_id=project_id,
                    collection_name=f"project_{project_id}",  # Defaulting to project-specific collection
                    doc_metadata=ingestion_metadata,
                )
                if success:
                    logger.info(
                        f"Successfully processed and ingested PDF '{safe_filename}' for project '{project_id}'."
                    )
                else:
                    logger.error(
                        f"Failed to process and ingest PDF '{safe_filename}' for project '{project_id}'. Check DocumentService logs."
                    )
                    # Potentially raise an HTTPException or alter response if ingestion failure is critical here
            else:
                logger.error(
                    f"Dependency injection for DocumentService is incorrect in upload_project_pdf. Type is {type(document_service)}"
                )
                # This indicates a setup issue with dependencies that needs to be resolved.

        except ImportError:
            logger.error(
                "Could not import DocumentService directly in upload_project_pdf. Dependency setup needed."
            )
        except Exception as ingest_e:
            logger.error(
                f"Error during PDF ingestion call for {safe_filename}: {ingest_e}",
                exc_info=True,
            )
            # Potentially raise an HTTPException or alter response

    except Exception as e:
        logger.error(
            f"Error saving uploaded PDF {safe_filename} to {file_path}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded PDF: {str(e)}",
        )
    finally:
        file.file.close()  # Ensure the file is closed

    return JSONResponse(
        content={
            "message": "File uploaded successfully.",
            "filename": safe_filename,
            "stored_path": str(
                file_path.relative_to(settings.PROJECT_DIR)
            ),  # Return path relative to project root
            "project_id": project_id,
        }
    )


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_deliverable_endpoint(
    request: GenerateDeliverableRequest,
    deliverable_service: DeliverableServiceDep,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """
    Triggers the generation of a deliverable based on a template and user inputs.
    This runs the section-by-section generation process.
    """
    start_time = time.time()
    project_id = request.project_id
    deliverable_type = request.deliverable_type
    user_id = current_user["id"]

    logger.info(
        f"Received request to generate deliverable '{deliverable_type}' for project '{project_id}' by user '{user_id}'."
    )

    # 1. Check Project Access
    if not project_manager.can_access_project(project_id, user_id):
        logger.warning(
            f"User {user_id} forbidden to generate deliverable for project {project_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this project.",
        )

    # 2. Call Deliverable Service (runs async)
    # We might run this as a background task later for long generations
    try:
        # Pass data from the request body
        result = await deliverable_service.generate_deliverable(
            project_id=project_id,
            deliverable_type=deliverable_type,
            user_json=request.user_input_data,
            conversation_id=request.conversation_id,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Deliverable generation call for '{deliverable_type}' project '{project_id}' completed in {processing_time:.2f}s with status: {result.get('status')}"
        )

        if result.get("status") == "success":
            # Return success response, maybe path to file or confirmation
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": f"Deliverable '{deliverable_type}' generated successfully.",
                    "project_id": project_id,
                    "deliverable_type": deliverable_type,
                    "output_path": result.get("markdown_path"),
                    "processing_time_seconds": processing_time,
                },
            )
        else:
            # Return error response from the service
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate deliverable: {result.get('error_message', 'Unknown error')}",
            )

    except Exception as e:
        logger.error(
            f"Error during deliverable generation endpoint call: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
