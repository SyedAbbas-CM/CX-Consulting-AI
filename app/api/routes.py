# app/api/routes.py
import asyncio
import glob
import json
import logging
import os
import shutil  # Added
import sys
import tempfile  # Added
import time
import uuid  # Added for P3 task_id
from datetime import datetime, timezone
from pathlib import Path
from tempfile import SpooledTemporaryFile  # Added
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks  # Re-add BackgroundTasks
from fastapi import Request  # <-- Add Request here
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,  # For RequestDisconnect
)

from app.agents.agent_runner import AgentRunner  # Ensure AgentRunner is imported
from app.api.auth import get_current_user
from app.api.dependencies import (
    AgentRunnerDep,
    ChatServiceDep,
    ContextOptimizerDep,
    CurrentUserDep,
    DeliverableServiceDep,
    DocumentServiceDep,
    LLMServiceDep,
    ProjectManagerDep,
    ProjectMemoryServiceDep,
    RagEngineDep,
    SettingsDep,
    TemplateManagerDep,
    TemplateServiceDep,
    get_document_service,
    get_project_manager,
    get_rag_engine,
)
from app.api.middleware.admin import admin_required
from app.api.models import OriginType  # Added OriginType import
from app.api.models import (  # Ensure this is imported, Added JobQueuedResponse
    ConversationInfo,
    ConversationsResponse,
    CXStrategyRequest,
    DeliverableResponse,
    DocumentGenerationConfig,
    DocumentUploadResponse,
    JobQueuedResponse,
    JourneyMapRequest,
    Project,
    ProjectCreateRequest,
    ProjectDocument,
    ProjectDocumentsResponse,
    ProjectsResponse,
    QuestionRequest,
    QuestionResponse,
    RefineRequest,
    ROIAnalysisRequest,
    SearchResult,
    Token,
    User,
    UserCreate,
)
from app.core.config import get_settings  # Added import
from app.core.llm_service import LLMService
from app.core.query_router import (  # Added QueryIntent
    QueryIntent,
    classify_query_intent,
)
from app.schemas.chat import (
    ChatCreateRequest,
    ChatCreateResponse,
    ChatHistoryResponse,
    ChatMessageCreateRequest,
    ChatMessageResponse,
    ChatSummaryResponse,
    RefinementResponse,
)

# Import schemas from their new locations
from app.schemas.model import LlmConfigResponse, ModelActionRequest
from app.scripts.model_manager import (
    MODELS_DIR,
    check_current_model,
    download_model,
    get_available_models,
    get_model_status,
    update_env_config,
)
from app.services import auth_service
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.ingest_worker import ingest_job  # Added
from app.services.project_manager import ProjectManager
from app.services.rag_engine import RagEngine
from app.utils.job_tracker import job_tracker  # Added

# Import from model_manager script
try:
    # Use absolute import from the app package root
    from app.scripts.model_manager import (
        AVAILABLE_MODELS,
        MODELS_DIR,
        check_current_model,
        download_model,
        get_available_models,
        get_model_status,
        update_env_config,
    )
except ImportError as e:
    logging.error(f"Failed to import from app.scripts.model_manager: {e}")
    # Define placeholders if import fails to avoid runtime errors on router setup
    AVAILABLE_MODELS = {}
    get_available_models = lambda: {}
    check_current_model = lambda: (None, None, None)
    MODELS_DIR = "models"
    get_model_status = lambda model_id: {
        "status": "error",
        "message": "model_manager not loaded",
    }

    def download_model(*args, **kwargs):
        raise NotImplementedError("model_manager.py not found or failed to import")

    def update_env_config(*args, **kwargs):
        raise NotImplementedError("model_manager.py not found or failed to import")


# Configure logger
logger = logging.getLogger("cx_consulting_ai.api.routes")

# Create router
router = APIRouter(prefix="/api")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# --- Global LLMService, ChatService instances and their getters were removed here ---
# --- They should be accessed via app.state through FastAPI dependencies ---


@router.post(
    "/documents", response_model=JobQueuedResponse, status_code=status.HTTP_202_ACCEPTED
)  # Changed response_model and status_code
async def upload_document(
    # Dependencies first
    rag_engine: RagEngineDep,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
    background_tasks: BackgroundTasks,  # Added
    # Required File param
    file: UploadFile = File(...),
    # Optional Form params
    project_id: Optional[str] = Form(None),
    is_global: bool = Form(False),
):
    "Upload a document, checking project access if project_id is provided."
    logger.info(
        f"Legacy upload document request for project '{project_id}' by user {current_user.get('id')} filename: {file.filename}"
    )

    # Settings for file size (consider moving to actual settings object)
    settings = (
        get_settings()
    )  # Get settings for MAX_UPLOAD_SIZE_PER_FILE and UPLOAD_DIR
    max_size_bytes = settings.MAX_UPLOAD_SIZE_PER_FILE

    legacy_temp_dir = settings.UPLOAD_DIR / "_legacy_temp"
    legacy_temp_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create directory before using it

    temp_file_path_obj: Optional[Path] = None

    try:
        if not project_id and not is_global:
            raise HTTPException(
                status_code=400,
                detail="Either project_id must be provided or document must be marked as global",
            )

        if project_id:
            project = await asyncio.to_thread(project_manager.get_project, project_id)
            if not project:
                raise HTTPException(
                    status_code=404, detail=f"Project {project_id} not found"
                )
            can_access = await asyncio.to_thread(
                project_manager.can_access_project, project_id, current_user["id"]
            )
            if not can_access:
                logger.warning(
                    f"User {current_user.get('id')} forbidden to upload document to project {project_id}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to upload documents to this project.",
                )

        # Create a temporary file that ingest_job can access.
        # ingest_job will copy this to its own managed directory.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f"_{file.filename}",
            dir=legacy_temp_dir,  # Use the created dir
        ) as real_temp_file:
            temp_file_path_obj = Path(real_temp_file.name)
            # Ensure the _legacy_temp directory exists # This line is now redundant
            # temp_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Streaming upload to temporary file: {temp_file_path_obj}")

            current_size = 0
            chunk_size = 65536
            while chunk := await file.read(chunk_size):
                current_size += len(chunk)
                if current_size > max_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: {current_size / (1024*1024):.2f} MB. Maximum size: {max_size_bytes / (1024*1024):.2f} MB",
                    )
                real_temp_file.write(chunk)
            logger.info(
                f"Finished streaming legacy upload. Total size: {current_size / (1024*1024):.2f} MB. Temp file: {temp_file_path_obj}"
            )

        if not temp_file_path_obj or not temp_file_path_obj.exists():
            logger.error(
                "Temporary file was not created or does not exist after streaming."
            )
            raise HTTPException(
                status_code=500, detail="Failed to create temporary file for upload."
            )

        # --- Start: New logic using ingest_job ---
        job_id = uuid.uuid4().hex
        user_id = current_user.get("id")  # Assuming current_user is a dict with an 'id'
        if not user_id:
            logger.error("Could not retrieve user ID for job tracking.")
            raise HTTPException(
                status_code=500, detail="Internal error: User ID not found."
            )

        job_tracker.start(
            job_id=job_id,
            user_id=user_id,
            total_files=1,
            project_id=project_id or "_global_",
        )
        job_tracker.set_processing(job_id)

        # ingest_job expects a folder_path containing the files.
        # For this legacy endpoint, we pass the parent of the single temp file.
        # And specific metadata about that one file.
        single_file_metadata = {
            "filename": file.filename,
            "path": str(temp_file_path_obj),  # Path to the actual temp file
            "content_type": file.content_type or "application/octet-stream",
        }

        # The ingest_job is responsible for its own staging and cleanup of files it processes.
        # The folder_path for ingest_job here points to where our single temp file resides.
        # ingest_job will look into this folder_path.
        background_tasks.add_task(
            ingest_job,
            job_id=job_id,
            project_id=project_id,  # Pass project_id, can be None if is_global
            folder_path=temp_file_path_obj.parent,  # The directory containing the temp file
            user_id=user_id,
            uploaded_files_metadata=[
                single_file_metadata
            ],  # List containing metadata for the one file
            is_global_upload=is_global,  # Pass the is_global flag
        )

        # Schedule the original temp file for deletion after the task is queued
        # It's copied by ingest_job if needed.
        def cleanup_temp_file(path_to_delete: Path):
            try:
                if path_to_delete.exists():
                    path_to_delete.unlink()
                    logger.info(f"Cleaned up legacy temp file: {path_to_delete}")
            except Exception as e_clean:
                logger.error(
                    f"Error cleaning up legacy temp file {path_to_delete}: {e_clean}"
                )

        background_tasks.add_task(cleanup_temp_file, temp_file_path_obj)

        logger.info(
            f"Legacy upload for {file.filename} converted to async job_id: {job_id}"
        )
        return JobQueuedResponse(
            job_id=job_id,
            message="File received and queued for processing via legacy endpoint.",
        )  # Changed to return JobQueuedResponse
        # --- End: New logic using ingest_job ---

    except HTTPException as http_exc:  # Specific re-raise for HTTPExceptions
        if temp_file_path_obj and temp_file_path_obj.exists():
            logger.debug(
                f"Cleaning up temporary upload file (HTTPException): {temp_file_path_obj}"
            )
            try:
                temp_file_path_obj.unlink()
            except Exception as e_clean_http:
                logger.error(
                    f"Error cleaning up temp file on HTTPException: {e_clean_http}"
                )
        raise http_exc
    except Exception as e:
        logger.error(
            f"Error in legacy upload_document for {file.filename}: {e}", exc_info=True
        )
        if temp_file_path_obj and temp_file_path_obj.exists():
                    logger.debug(
                f"Cleaning up temporary upload file (general exception): {temp_file_path_obj}"
            )
            try:
                temp_file_path_obj.unlink()
            except Exception as e_clean_exc:
                logger.error(
                    f"Error cleaning up temp file on general exception: {e_clean_exc}"
                )

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during file upload: {str(e)}",
        )
    # finally:
    # Original finally block for cleanup might be redundant if background task handles it,
    # or if exceptions above handle cleanup.
    # If temp_file_path was created and not passed to a cleanup task, it should be deleted here.
    # However, with the new logic, it's scheduled for cleanup by a background task.


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    question_request: QuestionRequest,
    http_request: Request,  # Add Request for disconnect check (G2)
    rag_engine: RagEngineDep,
    agent_runner: AgentRunnerDep,  # Added AgentRunnerDep
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    llm_service: LLMServiceDep,
    template_manager: TemplateManagerDep,
    project_memory_service: ProjectMemoryServiceDep,
    settings: SettingsDep,
):
    """Handles user questions, routes to RAG or direct LLM, handles timeouts and disconnects."""
    start_time = time.perf_counter()
    logger.info(
        f"Received question from user '{current_user.get('id')}' for project '{question_request.project_id}': '{question_request.query[:50]}...'"
    )

    # 0. Ensure project access (keep this outside the disconnect try/except)
    if question_request.project_id and not await asyncio.to_thread(
        project_manager.can_access_project,
        question_request.project_id,
        current_user["id"],
    ):
        logger.warning(
            f"User {current_user.get('id')} forbidden to ask question in project {question_request.project_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to ask questions in this project.",
        )

    response_data = {}
    conversation_id = question_request.conversation_id
    user_id = current_user["id"]  # Get user_id for create_document
    processing_time_start = (
        time.perf_counter()
    )  # More accurate start time for this specific path

    try:
        # --- Start main processing block --- 
        
        # Check for client disconnect before proceeding (G2)
        if await http_request.is_disconnected():
            logger.warning(
                f"Client disconnected before processing started for project '{question_request.project_id}'."
            )
            return JSONResponse(
                status_code=499, content={"detail": "Client closed request"}
            )

        # === New Conversational Deliverable Flow (Gemini O-3) ===
        if question_request.active_deliverable_type:
            logger.info(
                "Active deliverable flow triggered for type: %s",
                question_request.active_deliverable_type,
            )

            try:
                # ------- generate the document -------
                if not agent_runner:
                    raise HTTPException(
                        status_code=500, detail="AgentRunner not available."
                    )

                doc_dict = await agent_runner.generate_deliverable(
                    deliverable_type=question_request.active_deliverable_type,
                    user_turn=question_request.query,
                    project_id=question_request.project_id,
                    conversation_id=conversation_id,
                )

                # ------- add the user's turn to history -------
        if conversation_id:
            await chat_service.add_message_to_chat(
                chat_id=conversation_id,
                project_id=question_request.project_id,
                role="user",
                        content=question_request.query,
                        user_id=user_id,
                    )
                    logger.debug(
                        "User message stored in chat %s (deliverable).", conversation_id
                    )

                # ------- add the assistant's answer to history -------
                if conversation_id and doc_dict and doc_dict.get("content"):
                    await chat_service.add_message_to_chat(
                        chat_id=conversation_id,
                        project_id=question_request.project_id,
                        role="assistant",
                        content=(
                            f"Generated document: "
                            f"{doc_dict.get('title', question_request.active_deliverable_type)}\n\n"
                            f"{doc_dict['content'][:500]}… (full doc saved)"
                        ),
                        user_id="model",
                    )
                    logger.debug(
                        "Assistant (deliverable) message stored in chat %s.",
                        conversation_id,
                    )

                # ------- return immediately – no 'chat' mode work needed -------
                return QuestionResponse(
                    answer=doc_dict.get(
                        "content",
                        "Error: Document content not generated or found after generation.",
                    ),
                    conversation_id=conversation_id,
                    project_id=question_request.project_id,
                    sources=[],  # sources live in the doc
                    processing_time=time.perf_counter() - processing_time_start,
                )

            except Exception as e_gen_deliv:
                logger.error(
                    "generate_deliverable flow failed: %s", e_gen_deliv, exc_info=True
                )

                # still store the user's turn so the chat isn't broken
                if conversation_id:
                    await chat_service.add_message_to_chat(
                        chat_id=conversation_id,
                        project_id=question_request.project_id,
                        role="user",
                        content=question_request.query,
                        user_id=user_id,
                    )

                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate deliverable: {e_gen_deliv}",
                )
        # === End New Conversational Deliverable Flow ===

        # Original flow for /ask (chat mode, old document mode)
        # Add user message to chat history if not handled by deliverable flow
        if not question_request.active_deliverable_type and conversation_id:
            await chat_service.add_message_to_chat(
                chat_id=conversation_id,
                project_id=question_request.project_id,
                role="user",
                content=question_request.query,  # query is the main input for both modes
                user_id=current_user["id"],
            )
            logger.debug(
                f"User message (mode: {question_request.mode}) added to chat {conversation_id}"
            )
        else:
            logger.warning(
                f"No conversation_id provided for /ask endpoint (mode: {question_request.mode}). History not saved for user query."
            )

        # Check for disconnect again before long RAG/Agent call (G2)
        if await http_request.is_disconnected():
            logger.warning(
                f"Client disconnected before main processing for project '{question_request.project_id}'."
            )
            return JSONResponse(
                status_code=499, content={"detail": "Client closed request"}
            )

        # --- Workstream 5: Mode-based handling ---
        if question_request.mode == "document":
            if not question_request.active_deliverable_type:
                raise HTTPException(
                    status_code=400,
                    detail="active_deliverable_type is required for document mode.",
                )

            logger.info(
                f"Document generation mode for deliverable type: {question_request.active_deliverable_type}"
            )
            # This flow directly calls agent_runner.generate_deliverable
            generated_doc_dict = await agent_runner.generate_deliverable(
                deliverable_type=question_request.active_deliverable_type,
                user_turn=question_request.query,
                project_id=question_request.project_id,
                conversation_id=question_request.conversation_id,
            )
            if not generated_doc_dict or not generated_doc_dict.get("id"):
                raise HTTPException(
                    status_code=500, detail="Failed to generate or save document."
                )

            # Success: document was generated and persisted.
            # Return a summary message and the document ID.
            answer = f"Successfully generated document: '{generated_doc_dict.get('title', 'Untitled Document')}'"
            document_id = generated_doc_dict.get("id")
            sources = []  # No direct sources for this type of response message
            response_data.update(
                {
                    "answer": answer,
                    "document_id": document_id,
                    "sources": sources,
                    "conversation_id": question_request.conversation_id,  # Ensure conversation_id is passed back
                }
            )
            # Ensure project_id from request is also included if needed by QuestionResponse model
            # (It's not currently in QuestionResponse, but good to keep in mind)
            # response_data["project_id"] = question_request.project_id

        else:  # mode == "chat" or default
            logger.info(
                f"Processing /ask request in 'chat' mode for project '{question_request.project_id}'"
            )
            # 2. Classify intent (original logic for chat mode)
            intent = await asyncio.to_thread(
                classify_query_intent, question_request.query
            )
            query_for_engine = question_request.query
            logger.info(
                f"Query intent classified as: {intent} for question: '{query_for_engine[:50]}...'"
            )

            # 4. Handle based on intent
            if intent == QueryIntent.META_DOC_CHECK:
                logger.info(
                    f"Handling META_DOC_CHECK for project '{question_request.project_id}'"
                )
                if not question_request.project_id:
                    answer = "I need a project context to check for recently uploaded documents."
                    sources = []
                else:
                    latest_doc_summary = (
                        await project_manager.get_latest_document_summary(
                project_id=question_request.project_id,
                            user_id=current_user[
                                "id"
                            ],  # Pass user_id for potential filtering
                        )
                    )
                    if latest_doc_summary:
                        filename = latest_doc_summary.get("title", "Unknown Document")
                        # page_count and chunk_count might be in top-level or in metadata
                        # Let's check metadata first, then top-level as fallback.
                        doc_meta_for_counts = latest_doc_summary.get("metadata", {})
                        page_count = doc_meta_for_counts.get(
                            "page_count",
                            latest_doc_summary.get(
                                "page_count", "an unknown number of"
                            ),
                        )
                        chunk_count = doc_meta_for_counts.get(
                            "chunk_count",
                            latest_doc_summary.get(
                                "chunk_count", "an unknown number of"
                            ),
                        )

                        created_at_ts = latest_doc_summary.get("created_at")
                        upload_time_str = "at an unknown time"
                        if created_at_ts:
                            try:
                                upload_time_str = datetime.fromtimestamp(
                                    float(created_at_ts)
                                ).strftime("%H:%M on %Y-%m-%d")
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Could not parse timestamp '{created_at_ts}' (type: {type(created_at_ts)}) for META_DOC_CHECK: {e}"
                                )

                        doc_meta = latest_doc_summary.get("metadata", {})
                        subject_guess = doc_meta.get(
                            "extracted_title", doc_meta.get("title", filename)
                        )

                        answer = (
                            f'Yes, I see the document "{filename}" was uploaded for this project '
                            f"at {upload_time_str}. "
                            f"I have extracted {page_count} pages (resulting in {chunk_count} chunks of text). "
                            f'It appears to be about: "{subject_guess}". '
                            "You can now ask me questions about its content."
                        )
                        sources = [
                            {
                                "source": "project_document_confirmation",
                                "document_id": latest_doc_summary.get("id"),
                                "title": filename,
                            }
                        ]
                    else:
                        answer = "I couldn't find any recently uploaded documents for this project."
                        sources = []

                response_data.update(
                    {
                        "answer": answer,
                        "sources": sources,
                        "intent": QueryIntent.META_DOC_CHECK.value,
                    }
                )

            elif intent == QueryIntent.CHIT_CHAT or intent == QueryIntent.DIRECT:
                response_data = await rag_engine.ask(
                    question=query_for_engine,
                    project_id=question_request.project_id,
                    conversation_id=conversation_id,
                    user_id=current_user["id"],
                    retrieval_active=False,
                )
            # ... other intent handling ...
            else:  # Fallback for unknown or complex (RAG needed)
                logger.info(f"Handling RAG/QA for query: '{query_for_engine[:50]}...'")
                try:
                    rag_response = await rag_engine.ask(
                    question=query_for_engine,
                        conversation_id=question_request.conversation_id,
                    project_id=question_request.project_id,
                        user_id=user_id,
                        retrieval_active=True,
                    )

                    # Filter sources to only include project uploads
                    filtered_sources = []
                    raw_sources = rag_response.get("sources")
                    if raw_sources and isinstance(raw_sources, list):
                        for src_data in raw_sources:
                            # Ensure src_data is a dictionary before accessing .get()
                            if (
                                isinstance(src_data, dict)
                                and src_data.get("origin_type")
                                == OriginType.PROJECT_UPLOAD
                            ):
                                try:
                                    # Validate and convert to SearchResult model
                                    # This ensures that only valid SearchResult objects are passed forward
                                    filtered_sources.append(SearchResult(**src_data))
                                except Exception as e_pydantic:
                                    logger.warning(
                                        f"Could not form SearchResult from source data: {src_data}, error: {e_pydantic}"
                                    )
                            elif not isinstance(src_data, dict):
                                logger.warning(
                                    f"Encountered non-dict source item: {src_data}"
                                )
                    else:
                        if raw_sources is not None:  # Log if it exists but isn't a list
                            logger.warning(
                                f"Expected 'sources' to be a list, got: {type(raw_sources)}"
                            )

                    processing_time = (
                        time.time() - start_time
                    )  # Make sure start_time is defined at the beginning of ask_question
                    return QuestionResponse(
                        answer=rag_response["answer"],
                        sources=filtered_sources,  # Use filtered sources
                        conversation_id=rag_response.get("conversation_id"),
                        project_id=rag_response.get("project_id"),
                        processing_time=processing_time,
                    )
                except StarletteHTTPException as e:
                    logger.error(f"Error handling RAG/QA: {e}", exc_info=True)
                    raise HTTPException(
                        status_code=500, detail=f"Failed to handle RAG/QA: {str(e)}"
                )
        # --- End Workstream 5 mode-based handling ---
            
        # Check for disconnect *after* RAG/Agent call finishes (G2)
        if await http_request.is_disconnected():
            logger.warning(
                f"Client disconnected after main processing for project '{question_request.project_id}'. Result discarded."
            )
            # Don't proceed to save history or return data
            return JSONResponse(
                status_code=499, content={"detail": "Client closed request"}
            )

        # 5. Add assistant message to chat history (if response is valid and not timed out)
        if (
            conversation_id
            and response_data
            and response_data.get("answer")
            and response_data.get("error") != "timeout"
        ):
            await chat_service.add_message_to_chat(
                chat_id=conversation_id,
                project_id=question_request.project_id,
                role="assistant",
                content=response_data["answer"],
                user_id=current_user["id"],  # Or a system/bot ID
            )
            logger.debug(f"Assistant message added to chat {conversation_id}")

            # 6. Append to project memory (if response is valid and not timed out)
            if (
                settings.ENABLE_PROJECT_MEMORY
                and question_request.project_id
                and response_data.get("error") != "timeout"
            ):
                try:
                    await project_memory_service.append_raw_interaction(
                        project_id=question_request.project_id,
                        user_query=question_request.query,
                        llm_response=response_data["answer"],
                    )
                    logger.info(
                        f"Successfully appended interaction to memory for project {question_request.project_id}"
                    )
                except Exception as mem_e:
                    logger.error(
                        f"Failed to append interaction to memory for project {question_request.project_id}: {mem_e}",
                        exc_info=True,
                    )

    # Handle potential disconnect during the main block
    except StarletteHTTPException as http_exc:  # Catch Starlette disconnect
        if http_exc.status_code == 499:  # Check if it's a client disconnect
            logger.warning(
                f"Client disconnected during processing for project '{question_request.project_id}'."
            )
            return JSONResponse(
                status_code=499, content={"detail": "Client closed request"}
            )
        else:
            # Re-raise other HTTP exceptions
            raise http_exc
    except KeyError as e:
        logger.error(f"KeyError: {e}", exc_info=True)
        response_data = {
            "answer": f"Internal error: Missing expected data key: {e}",
            "sources": [],
            "intent": "internal_key_error",
        }
    except Exception as e:
        logger.error(f"Unhandled error in /ask endpoint: {e}", exc_info=True)
        response_data = {
            "answer": "An unexpected server error occurred.",
            "sources": [],
            "intent": "unhandled_endpoint_error",
            # Ensure conversation_id is a string, even in error cases, if the model expects it.
            # If conversation_id was not established before the error, use a placeholder or empty string.
            "conversation_id": conversation_id if conversation_id is not None else "", 
            "project_id": (
                question_request.project_id if question_request else "unknown_project"
            ),
        }

    # --- End main processing block ---

    # Final response handling:
    processing_time_calculated = time.perf_counter() - start_time
    
    # Check for timeout error from RAG engine (G1)
    if response_data.get("error") == "timeout":
        logger.warning(
            f"Request timed out for project '{question_request.project_id}'. Returning 504."
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "detail": response_data.get(
                    "answer", "Request timed out during processing."
                ),
                "processing_time": processing_time_calculated,
            },
        )
        
    # Check for template rendering error from RAG engine (G3)
    if response_data.get("error") == "template_render":
        logger.warning(
            f"Template rendering error for project '{question_request.project_id}'. Returning 422."
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": response_data.get(
                    "answer", "Failed to process request due to template error."
                ),
                "missing_key_info": response_data.get(
                    "detail", "Unknown key"
                ),  # Include detail from RagEngine
                "processing_time": processing_time_calculated,
            },
        )

    # Populate mandatory fields if not already set by error handlers
    response_data.setdefault("conversation_id", conversation_id)
    response_data.setdefault("project_id", question_request.project_id)
    response_data.setdefault("processing_time", processing_time_calculated)
    response_data.setdefault(
        "sources", response_data.get("sources", [])
    )  # Ensure sources is always a list
    response_data.setdefault(
        "answer", response_data.get("answer", "Sorry, I encountered an issue.")
    )

    # Remove internal error flag before returning successful response
    response_data.pop("error", None)
    response_data.pop("detail", None)  # Also remove detail field if it exists

    logger.info(
        f"Successfully processed question in {processing_time_calculated:.2f}s for project '{question_request.project_id}'"
    )
    logger.debug(f"Final response data for /ask: {response_data}")
    return QuestionResponse(**response_data)


@router.post(
    "/cx-strategy", response_model=DeliverableResponse
)  # P3: status_code changed if not background
async def generate_cx_strategy(
    request: CXStrategyRequest,
    agent_runner: AgentRunnerDep,  # Changed from rag_engine
    project_manager: ProjectManagerDep,  # Added
    current_user: CurrentUserDep,
    # background_tasks: BackgroundTasks, # P3: Removed BackgroundTasks
):
    user_id = current_user.get("id")
    project_id = request.project_id
    start_time = time.time()  # For processing_time

    if not project_id:
        raise HTTPException(
            status_code=400, detail="project_id is required for CX Strategy"
        )

    logger.info(
        f"CX strategy generation requested for project {project_id} by user {user_id} via AgentRunner."
    )

    try:
        # Construct DocumentGenerationConfig
        # The deliverable_type for doc_config might differ from task_type if there's a mapping
        # For now, assume they are the same. The literal for CXStrategyRequest could be 'cx_strategy' or 'proposal'.
        # Let's use 'cx_strategy' to match the endpoint and task_type.
        doc_gen_config = DocumentGenerationConfig(
            deliverable_type="cx_strategy", 
            parameters=request.model_dump(
                exclude={"project_id", "conversation_id", "query"}
            ),  # query might not exist on this model
        )

        agent_result = await agent_runner.run(
            task_type="cx_strategy", 
            query=json.dumps(
                request.model_dump()
            ),  # Pass full request as JSON string for now
            project_id=project_id,
            conversation_id=request.conversation_id,  # Pass conversation_id
            doc_config=doc_gen_config,
        )

        final_content = agent_result.get(
            "final_answer", "Error: Content not generated."
        )
        # sources = agent_result.get("sources", []) # DeliverableResponse doesn't have sources

        # Persist the document
        persisted_doc_id = await project_manager.create_document(
            project_id=project_id,
            user_id=user_id, 
            title=f"CX Strategy for {request.client_name}", 
            content=final_content,
            document_type="cx_strategy", 
            metadata={
                "generated_by": "AgentRunner",
                "task_type": "cx_strategy",
                "parameters": doc_gen_config.parameters,
                "original_request": request.model_dump(),
            },
        )
        
        processing_time = time.time() - start_time
        return DeliverableResponse(
            content=final_content,
            conversation_id=request.conversation_id
            or str(uuid.uuid4()),  # Ensure conversation_id
            project_id=project_id,
            document_id=persisted_doc_id,
            processing_time=processing_time,
        )

    except Exception as e_task:
        logger.error(
            f"AgentRunner task failed for CX strategy {project_id}: {e_task}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to generate CX strategy: {str(e_task)}"
        )


@router.post("/roi-analysis", response_model=DeliverableResponse)
async def generate_roi_analysis(
    request: ROIAnalysisRequest,
    agent_runner: AgentRunnerDep,  # Changed
    project_manager: ProjectManagerDep,  # Added
    current_user: CurrentUserDep,
    # background_tasks: BackgroundTasks, # Removed
):
    user_id = current_user.get("id")
    project_id = request.project_id
    start_time = time.time()

    if not project_id:
        raise HTTPException(
            status_code=400, detail="project_id is required for ROI Analysis"
        )

    logger.info(
        f"ROI analysis generation requested for project {project_id} by user {user_id} via AgentRunner."
    )
    
    try:
        doc_gen_config = DocumentGenerationConfig(
            deliverable_type="roi_analysis", 
            parameters=request.model_dump(exclude={"project_id", "conversation_id"}),
        )

        agent_result = await agent_runner.run(
            task_type="roi_analysis", 
            query=json.dumps(request.model_dump()),
            project_id=project_id,
            conversation_id=request.conversation_id,  # Pass conversation_id
            doc_config=doc_gen_config,
        )

        final_content = agent_result.get(
            "final_answer", "Error: Content not generated."
        )

        persisted_doc_id = await project_manager.create_document(
            project_id=project_id,
            user_id=user_id,
            title=f"ROI Analysis for {request.client_name}",
            content=final_content,
            document_type="roi_analysis",
            metadata={
                "generated_by": "AgentRunner",
                "task_type": "roi_analysis",
                "parameters": doc_gen_config.parameters,
                "original_request": request.model_dump(),
            },
        )
        
        processing_time = time.time() - start_time
        return DeliverableResponse(
            content=final_content,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            project_id=project_id,
            document_id=persisted_doc_id,
            processing_time=processing_time,
        )

    except Exception as e_task:
        logger.error(
            f"AgentRunner task failed for ROI analysis {project_id}: {e_task}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to generate ROI analysis: {str(e_task)}"
        )


@router.post("/journey-map", response_model=DeliverableResponse)
async def generate_journey_map(
    request: JourneyMapRequest,
    agent_runner: AgentRunnerDep,  # Changed
    project_manager: ProjectManagerDep,  # Added
    current_user: CurrentUserDep,
    # background_tasks: BackgroundTasks, # Removed
):
    user_id = current_user.get("id")
    project_id = request.project_id
    start_time = time.time()

    if not project_id:
        raise HTTPException(
            status_code=400, detail="project_id is required for Journey Map"
        )

    logger.info(
        f"Journey map generation requested for project {project_id} by user {user_id} via AgentRunner."
    )

    try:
        doc_gen_config = DocumentGenerationConfig(
            deliverable_type="journey_map", 
            parameters=request.model_dump(exclude={"project_id", "conversation_id"}),
        )

        agent_result = await agent_runner.run(
            task_type="journey_map", 
            query=json.dumps(request.model_dump()),
            project_id=project_id,
            conversation_id=request.conversation_id,  # Pass conversation_id
            doc_config=doc_gen_config,
        )

        final_content = agent_result.get(
            "final_answer", "Error: Content not generated."
        )

        persisted_doc_id = await project_manager.create_document(
            project_id=project_id,
            user_id=user_id,
            title=f"Journey Map for {request.client_name} ({request.persona})",
            content=final_content,
            document_type="journey_map",
            metadata={
                "generated_by": "AgentRunner",
                "task_type": "journey_map",
                "parameters": doc_gen_config.parameters,
                "original_request": request.model_dump(),
            },
        )
        
        processing_time = time.time() - start_time
        return DeliverableResponse(
            content=final_content,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            project_id=project_id,
            document_id=persisted_doc_id,
            processing_time=processing_time,
        )

    except Exception as e_task:
        logger.error(
            f"AgentRunner task failed for Journey map {project_id}: {e_task}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to generate journey map: {str(e_task)}"
        )


@router.get("/conversations", response_model=ConversationsResponse)
async def get_conversations(
    # Dependencies first (no required path/body)
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    # Optional Query/Path params
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    project_id: Optional[str] = None,
):
    """Get all conversations, optionally filtered by project_id."""
    logger.info(
        f"Get conversations request by user {current_user.get('id')}, project_id: {project_id}, limit: {limit}, offset: {offset}"
    )
    try:
        # Updated to use chat_service
        conversations_data = await chat_service.get_conversations_for_user(
            user_id=current_user.get("id"),
            project_id=project_id,
            limit=limit,
            offset=offset,
        )

        # Ensure the response matches ConversationsResponse model
        # This might require adapting the output of chat_service.get_conversations_for_user
        # or adjusting the ConversationsResponse model.
        # For now, assuming conversations_data is a list of dicts/objects that can be
        # directly used or slightly adapted.

        # Example adaptation if chat_service returns a list of ChatSummaryResponse-like objects
        infos = []
        for conv_summary in conversations_data.get(
            "summaries", []
        ):  # Assuming chat_service returns a dict with 'summaries' and 'total'
            infos.append(
                ConversationInfo(
                    id=conv_summary.id,
                    name=conv_summary.name,
                    created_at=conv_summary.created_at,
                    project_id=conv_summary.project_id,  # Assuming project_id is part of the summary
                    # last_message_at might need to be fetched or is part of summary
                    last_message_at=conv_summary.updated_at,  # Or a more specific field
                )
            )

        return ConversationsResponse(
            conversations=infos,
            total=conversations_data.get("total", 0),
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching conversations for user {current_user.get('id')}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    chat_service: ChatServiceDep,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Delete a specific conversation."""
    user_id = current_user.get("id")
    logger.info(
        f"Delete conversation request for conversation_id '{conversation_id}' by user {user_id}"
    )
    try:
        # First, check if the user has access to the project this conversation belongs to.
        # This logic might be inside chat_service.delete_conversation if it handles project ownership.
        # Or, we might need to fetch conversation details first to get project_id.

        # Assuming chat_service.delete_conversation handles permissions or we do it here:
        # chat_details = await chat_service.get_chat_summary(chat_id=conversation_id, user_id=user_id)
        # if not chat_details:
        #     raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        # if chat_details.project_id:
        #     if not await project_manager.user_has_access_to_project(user_id, chat_details.project_id):
        #         raise HTTPException(status_code=403, detail="User does not have access to the project this conversation belongs to")

        success = await chat_service.delete_chat_session(
            chat_id=conversation_id, user_id=user_id
        )
        if not success:
            # This could mean not found or not authorized, chat_service should clarify
            raise HTTPException(
                status_code=404, detail="Conversation not found or could not be deleted"
            )
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content={
                "message": "Conversation deleted successfully"
            },  # Content for 204 is usually empty
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting conversation {conversation_id} for user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@router.get("/health", tags=["Health"], response_model=Dict[str, Any])
async def health_check(
    chat_service: ChatServiceDep,
    # Add LLMServiceDep
    llm_service: LLMServiceDep,
):
    """Health check endpoint that verifies the status of all system components."""
    import time

    start_time = time.time()
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [],
    }
    overall_status = "healthy"

    # Check Redis via ChatService
    redis_status = "unhealthy"
    redis_details = "Connection failed or ping timeout"
    try:
        # Ping Redis via the client used by ChatService
        if await chat_service.redis_client.ping():
            redis_status = "healthy"
            redis_details = "Connected and responsive"
    except Exception as e:
        logger.error(f"Health check: Redis ping failed: {e}")
        redis_details = f"Connection failed: {e}"

    health_status["checks"].append(
        {
            "service": "redis (via ChatService)",
            "status": redis_status,
            "details": redis_details,
        }
    )
    if redis_status == "unhealthy":
        overall_status = "unhealthy"

    # Check LLM Service status (without generation test to avoid timeouts)
    llm_status = "healthy" if llm_service else "unhealthy"
    llm_details = (
        "LLM service available" if llm_service else "LLM service not initialized"
    )

    health_status["checks"].append(
        {
            "service": "llm_service",
            "status": llm_status,
            "details": llm_details,
            "backend": getattr(llm_service, "backend", "unknown"),
            "model_id": getattr(llm_service, "model_id", "unknown"),
        }
    )

    health_status["status"] = overall_status
    health_status["check_duration_ms"] = (time.time() - start_time) * 1000

    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status
        )

    return health_status


# Add a detailed health endpoint for LLM testing
@router.get("/health/detailed", tags=["Health"], response_model=Dict[str, Any])
async def detailed_health_check(
    chat_service: ChatServiceDep,
    llm_service: LLMServiceDep,
):
    """Detailed health check that includes LLM generation test (slower)."""
    import time

    start_time = time.time()
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [],
    }
    overall_status = "healthy"

    # Check Redis via ChatService
    redis_status = "unhealthy"
    redis_details = "Connection failed or ping timeout"
    try:
        if await chat_service.redis_client.ping():
            redis_status = "healthy"
            redis_details = "Connected and responsive"
    except Exception as e:
        logger.error(f"Health check: Redis ping failed: {e}")
        redis_details = f"Connection failed: {e}"

    health_status["checks"].append(
        {
            "service": "redis (via ChatService)",
            "status": redis_status,
            "details": redis_details,
        }
    )
    if redis_status == "unhealthy":
        overall_status = "unhealthy"

    # Check LLM Service (with generation test)
    llm_status = "unhealthy"
    llm_details = "LLM generation failed or timed out"
    try:
        # Simple test prompt
        test_prompt = "Health check prompt: respond OK."
        # Use a short timeout for the health check
        response = await asyncio.wait_for(
            llm_service.generate(prompt=test_prompt, max_tokens=5), timeout=10.0
        )
        if response and "OK" in response:
            llm_status = "healthy"
            llm_details = "LLM generated test response successfully"
        else:
            llm_details = f"LLM generated unexpected response: {response[:50]}..."

    except asyncio.TimeoutError:
        logger.error("Health check: LLM generation timed out.")
        llm_details = "LLM generation timed out after 10 seconds."
    except Exception as e:
        logger.error(f"Health check: LLM generation failed: {e}", exc_info=True)
        llm_details = f"LLM generation failed: {e}"

    health_status["checks"].append(
        {
            "service": "llm_service",
            "status": llm_status,
            "details": llm_details,
            "backend": llm_service.backend,
            "model_id": llm_service.model_id,
        }
    )
    if llm_status == "unhealthy":
        overall_status = "degraded" if overall_status == "healthy" else overall_status

    health_status["status"] = overall_status
    health_status["check_duration_ms"] = (time.time() - start_time) * 1000

    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status
        )

    return health_status


# PROJECTS API ENDPOINTS


@router.post("/projects", response_model=Project)
async def create_project(
    project_request: ProjectCreateRequest,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Creates a new project."""
    try:
        # P6: Wrap sync ProjectManager call
        project_id = await asyncio.to_thread(
            project_manager.create_project,
            name=project_request.name,
            client_name=project_request.client_name,
            industry=project_request.industry,
            description=project_request.description,
            owner_id=current_user["id"],
            shared_with=project_request.shared_with,
            metadata=project_request.metadata,
        )
        # P6: Wrap sync ProjectManager call
        project_data = await asyncio.to_thread(project_manager.get_project, project_id)
        if not project_data:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve project after creation"
            )
        return Project(**project_data)
    except Exception as e:
        logger.error(
            f"Unexpected error creating project '{project_request.name}': {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while creating the project.",
        )


@router.get("/projects", response_model=ProjectsResponse)
async def list_projects(
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """List all projects accessible by the current user (owned or shared)."""
    try:
        # P6: Wrap sync ProjectManager call
        # FIX for ValueError: too many values to unpack
        projects_result = await asyncio.to_thread(
            project_manager.get_user_projects, current_user["id"]
        )
        if isinstance(
            projects_result, tuple
        ):  # Handle if function returns (list, count) tuple
            projects_data, total_count = projects_result
        else:  # Handle if function returns just the list
            projects_data = projects_result 
            total_count = len(projects_data) if projects_data else 0
            
        return ProjectsResponse(
            projects=[Project(**p) for p in projects_data], count=total_count
        )
    except Exception as e:
        # ... (existing error handling) ...
        logger.error(
            f"Error listing projects for user {current_user.get('id')}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")


@router.get("/projects/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Get a project by ID, checking access."""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            # Keep 404 if project doesn't exist at all
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )

        # Add access check
        if not project_manager.can_access_project(project_id, current_user["id"]):
            logger.warning(
                f"User {current_user.get('id')} forbidden access to project {project_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this project.",
            )

        return project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting project {project_id} for user {current_user.get('id')}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Error getting project: {str(e)}")


@router.put("/projects/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    updates: ProjectCreateRequest,  # Use the Pydantic model for updates
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Update a project. Only the owner can update."""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )

        # Check ownership
        if project.get("owner_id") != current_user.get("id"):
            raise HTTPException(
                status_code=403, detail="Only the project owner can update the project."
            )

        # Convert Pydantic model to dict, excluding unset fields to avoid overwriting with None
        update_data = updates.model_dump(exclude_unset=True)

        success = project_manager.update_project(project_id, update_data)
        if not success:
            # This might indicate a race condition or internal error if the project existed moments ago
            raise HTTPException(
                status_code=500, detail=f"Failed to update project {project_id}"
            )

        # Return the updated project
        updated_project = project_manager.get_project(project_id)
        if not updated_project:
            raise HTTPException(
                status_code=404,
                detail=f"Updated project {project_id} could not be retrieved.",
            )
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
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
):
    """Delete a project and its associated chats. Only the owner can delete."""
    logger.info(
        f"Attempting to delete project {project_id} by user {current_user.get('id')}"
    )
    try:
        # 1. Check if project exists
        project = project_manager.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for deletion.")
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )

        # 2. Check ownership
        if project.get("owner_id") != current_user.get("id"):
            logger.warning(
                f"User {current_user.get('id')} attempted to delete project {project_id} owned by {project.get('owner_id')}"
            )
            raise HTTPException(
                status_code=403, detail="Only the project owner can delete the project."
            )

        # 3. Get associated chat IDs BEFORE deleting the project
        # Assuming project_manager stores chat IDs, or we need to query chat_service
        # Let's assume ProjectManager holds the list
        chat_ids_to_delete = project.get(
            "conversation_ids", []
        )  # Use 'conversation_ids' as per Project model
        logger.info(
            f"Found {len(chat_ids_to_delete)} chats associated with project {project_id} for deletion: {chat_ids_to_delete}"
        )

        # 4. Delete associated chats
        deleted_chat_count = 0
        failed_chat_deletions = []
        for chat_id in chat_ids_to_delete:
            try:
                # Use the new delete_chat endpoint logic (or call chat_service directly)
                success = await chat_service.delete_chat(chat_id)
                if success:
                    deleted_chat_count += 1
                    logger.info(
                        f"Deleted associated chat {chat_id} for project {project_id}"
                    )
                else:
                    # This might happen if the chat was already deleted or doesn't exist
                    logger.warning(
                        f"Could not delete associated chat {chat_id} for project {project_id} (might not exist)."
                    )
                    failed_chat_deletions.append(chat_id)
            except Exception as chat_del_err:
                logger.error(
                    f"Error deleting associated chat {chat_id} for project {project_id}: {chat_del_err}",
                    exc_info=True,
                )
                failed_chat_deletions.append(chat_id)

        if failed_chat_deletions:
            # Decide how to handle partial failures. For now, log a warning and proceed with project deletion.
            logger.warning(
                f"Failed to delete some associated chats for project {project_id}: {failed_chat_deletions}"
            )

        # 5. Delete the project itself
        success = project_manager.delete_project(project_id)
        if not success:
            # Should not happen if checks passed, but handle defensively
            logger.error(
                f"Project manager failed to delete project {project_id} after ownership check."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete project {project_id} after deleting chats.",
            )

        logger.info(
            f"Successfully deleted project {project_id} and {deleted_chat_count} associated chats."
        )
        return None  # Return None for 204 No Content
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")


# PROJECT DOCUMENTS API ENDPOINTS


@router.get("/projects/{project_id}/documents", response_model=ProjectDocumentsResponse)
async def get_project_documents(
    project_id: str,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Get all documents for a project, checking access."""
    logger.info(
        f"Request for documents in project {project_id} by user {current_user.get('id')}"
    )
    try:
        # Check if project exists and user has access
        # Use can_access_project which implicitly checks existence via get_project
        if not project_manager.can_access_project(project_id, current_user["id"]):
            # Distinguish between not found and forbidden
            project_exists = project_manager.get_project(project_id) is not None
            if not project_exists:
                logger.warning(
                    f"Project {project_id} not found when listing documents."
                )
                raise HTTPException(
                    status_code=404, detail=f"Project {project_id} not found"
                )
            else:
                logger.warning(
                    f"User {current_user.get('id')} forbidden to list documents for project {project_id}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to access documents in this project.",
                )

        # If access is granted, get the documents
        documents = project_manager.get_project_documents(project_id)
        return ProjectDocumentsResponse(documents=documents, count=len(documents))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting documents for project {project_id}, user {current_user.get('id')}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error getting project documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=ProjectDocument)
async def get_document(
    document_id: str,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Get a document by ID, checking access via its project."""
    logger.info(f"Request for document {document_id} by user {current_user.get('id')}")
    try:
        document = project_manager.get_document(document_id)
        if not document or not document.get("project_id"):
            logger.warning(f"Document {document_id} not found or missing project_id.")
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found or invalid."
            )

        project_id = document["project_id"]

        # Check access to the project this document belongs to
        if not project_manager.can_access_project(project_id, current_user["id"]):
            logger.warning(
                f"User {current_user.get('id')} forbidden to access document {document_id} (project {project_id})."
            )
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this document.",
            )

        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting document {document_id} for user {current_user.get('id')}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")


@router.put("/documents/{document_id}", response_model=ProjectDocument)
async def update_document(
    document_id: str,
    updates: dict,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """Update a document, checking access via its project."""
    logger.info(
        f"Request to update document {document_id} by user {current_user.get('id')}"
    )
    try:
        # 1. Get document and check existence
        document = project_manager.get_document(document_id)
        if not document or not document.get("project_id"):
            logger.warning(
                f"Document {document_id} not found or missing project_id for update."
            )
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found or invalid."
            )

        project_id = document["project_id"]

        # 2. Check access to the project
        if not project_manager.can_access_project(project_id, current_user["id"]):
            logger.warning(
                f"User {current_user.get('id')} forbidden to update document {document_id} (project {project_id})."
            )
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to update this document.",
            )

        # 3. Perform update
        success = project_manager.update_document(document_id, updates)
        if not success:
            # This might happen if the document was deleted between check and update
            logger.error(f"Failed to update document {document_id} after access check.")
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found during update.",
            )

        # Return the updated document
        updated_document = project_manager.get_document(document_id)
        if not updated_document:
            # Should not happen if update was successful
            logger.error(
                f"Failed to retrieve document {document_id} after successful update."
            )
            raise HTTPException(
                status_code=500, detail="Failed to retrieve document after update."
            )
        return updated_document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error updating document {document_id} for user {current_user.get('id')}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error updating document: {str(e)}"
        )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],  # Add to Documents tag group
    summary="Delete a document and its chunks, checking access",
)
async def delete_document_route(
    document_id: str,
    doc_service: DocumentServiceDep,
    project_manager: ProjectManagerDep,
    current_user: CurrentUserDep,
):
    """
    Deletes a document and its associated chunks from the vector store,
    after verifying the user has access to the document's project.
    """
    logger.info(
        f"Received request to delete document {document_id} by user {current_user.get('id')}"
    )
    try:
        # 1. Get document details to find project ID
        document = project_manager.get_document(document_id)
        if not document or not document.get("project_id"):
            # If document doesn't exist, treat as success (idempotent delete)
            logger.warning(
                f"Document {document_id} not found or invalid for deletion. Assuming already deleted."
            )
            return None  # Return 204 No Content

        project_id = document["project_id"]

        # 2. Check access to the project
        if not project_manager.can_access_project(project_id, current_user["id"]):
            logger.warning(
                f"User {current_user.get('id')} forbidden to delete document {document_id} (project {project_id})."
            )
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to delete this document.",
            )

        # 3. Perform the deletion using DocumentService (vector store)
        success = await doc_service.delete_document(document_id)
        if not success:
            # Log error but don't necessarily fail if vector store deletion had issues,
            # as we still want to remove it from the project manager.
            logger.error(
                f"Document service failed to process vector store deletion for ID: {document_id}"
            )
            # Raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            #                     detail=f"Failed to delete document '{document_id}' chunks. Check server logs.")

        # 4. Remove document reference from ProjectManager (even if vector deletion failed)
        # Use ProjectManager's delete method which handles file/redis removal and project list update
        pm_delete_success = project_manager.delete_document(document_id)
        if not pm_delete_success:
            # This shouldn't happen if get_document worked, but log defensively
            logger.error(
                f"Project manager failed to remove reference for document {document_id} after access check."
            )
            # Consider if this should be a 500 error

        logger.info(
            f"Successfully processed deletion request for document ID: {document_id} (Project Manager reference removed)."
        )
        return None  # Return None for 204 response

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            f"Error in delete document route for ID {document_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while deleting document '{document_id}'.",
        )


# PROJECT CONVERSATIONS API ENDPOINTS


@router.get(
    "/projects/{project_id}/conversations", response_model=ConversationsResponse
)
async def get_project_conversations(
    project_id: str,
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    limit: int = Query(50, ge=1, le=100),  # Added limit/offset
    offset: int = Query(0, ge=0),  # Added limit/offset
):
    """Get all conversations for a specific project."""
    user_id = current_user.get("id")
    logger.info(
        f"Get conversations for project '{project_id}' by user {user_id}, limit: {limit}, offset: {offset}"
    )

    # Check if user has access to the project
    if not await project_manager.user_has_access_to_project(user_id, project_id):
        raise HTTPException(
            status_code=403, detail="User does not have access to this project"
        )

    try:
        conversations_data = await chat_service.get_conversations_for_user(
            user_id=user_id, project_id=project_id, limit=limit, offset=offset
        )

        infos = []
        for conv_summary in conversations_data.get("summaries", []):
            infos.append(
                ConversationInfo(
                    id=conv_summary.id,
                    name=conv_summary.name,
                    created_at=conv_summary.created_at,
                    project_id=conv_summary.project_id,
                    last_message_at=conv_summary.updated_at,
                )
            )

        return ConversationsResponse(
            conversations=infos,
            total=conversations_data.get("total", 0),
            limit=limit,
            offset=offset,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching conversations for project {project_id} by user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to fetch project conversations"
        )


@router.post("/projects/{project_id}/associate-conversation/{conversation_id}")
async def associate_conversation_with_project(
    project_id: str,
    conversation_id: str,
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
):
    """Associate an existing conversation with a project."""
    user_id = current_user.get("id")
    logger.info(
        f"Associate conversation '{conversation_id}' with project '{project_id}' by user {user_id}"
    )

    # Check if user has access to the project
    if not await project_manager.user_has_access_to_project(user_id, project_id):
        raise HTTPException(
            status_code=403, detail="User does not have access to this project"
        )

    try:
        # The logic to associate might be in ChatService or ProjectManager
        # For ChatService, it might involve updating the conversation's project_id
        # Ensure the conversation is not already associated with another project or handle as needed.
        success = await chat_service.associate_chat_with_project(
            chat_id=conversation_id, project_id=project_id, user_id=user_id
        )

        if not success:
            # This could be due to various reasons: conversation not found, already associated, etc.
            # The service method should ideally raise specific exceptions or return more info.
            raise HTTPException(
                status_code=400,
                detail="Failed to associate conversation with project. Conversation may not exist, user may not own it, or it might already be associated.",
            )

        return {
            "message": "Conversation successfully associated with project",
            "project_id": project_id,
            "conversation_id": conversation_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error associating conversation {conversation_id} with project {project_id} for user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while associating the conversation.",
        )


# ADMIN ENDPOINTS


@router.get("/admin/users")
@admin_required
async def list_users(
    # Dependencies first
    current_user: CurrentUserDep,
    # Optional Query params
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
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
async def admin_update_user(user_id: str, updates: dict, current_user: CurrentUserDep):
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
    user_id: str, current_user: CurrentUserDep  # Required Path param  # Dependency
):
    """Delete a user (admin only)."""
    try:
        from app.services.auth_service import AuthService

        auth_service = AuthService()

        # Correct logic: Delete the user
        success = auth_service.delete_user(user_id)

        if not success:
            # Deletion might fail if user not found
                raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or could not be deleted.",
            )

        # Return No Content on successful deletion
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")


@router.get("/improvement/interactions", response_model=Dict[str, List[Dict[str, Any]]])
async def get_improvement_interactions(
    # Dependencies first
    current_user: CurrentUserDep,
    # Optional Query params
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get saved interactions for model improvement."""
    try:
        # Only admins can access this endpoint
        if not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this resource",
            )

        # Get the improvement directory
        improvement_dir = os.path.join("app", "data", "improvement")

        # Create directory if it doesn't exist
        os.makedirs(improvement_dir, exist_ok=True)

        # Get all JSON files in the directory
        files = sorted(
            glob.glob(os.path.join(improvement_dir, "*.json")),
            key=os.path.getmtime,
            reverse=True,
        )

        # Apply pagination
        paginated_files = files[offset : offset + limit]

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
        raise HTTPException(
            status_code=500, detail=f"Error retrieving interactions: {str(e)}"
        )


# --- NEW Model Management Routes ---


@router.get("/models", tags=["Models"])
async def list_available_models(current_user: CurrentUserDep):
    """Lists available models and their download status."""
    models_status_list = []
    # Use the imported function that handles potential import errors
    available_models_dict = get_available_models()

    if not available_models_dict:
        logger.warning(
            "AVAILABLE_MODELS dictionary is empty or model_manager failed to load."
        )
        # Return empty list or appropriate error response?
        # Let's return empty for now, but ideally we'd know if it failed vs. was just empty.
        return {"available_models": [], "active_model_path": "Unknown"}

    try:
        for model_id, info in available_models_dict.items():
            # Get status for each model
            status_info = get_model_status(model_id)

            models_status_list.append(
                {
                    # Base info from AVAILABLE_MODELS
                    "id": model_id,
                    "name": info.get(
                        "description", model_id
                    ),  # Use description as name, fallback to id
                    "description": info.get("description"),
                    "size_gb": info.get("size_gb"),
                    # Status info from get_model_status
                    "status": status_info.get("status", "unknown"),
                    "message": status_info.get("message", ""),
                    "path": status_info.get("path"),  # Include path if available
                    # Add a simple boolean for frontend convenience
                    "downloaded": status_info.get("status") == "available",
                }
            )

        # Get currently configured model path from .env (using check_current_model)
        _, active_filename, _ = check_current_model()
        active_model_path = (
            os.path.join(MODELS_DIR, active_filename) if active_filename else "Unknown"
        )

        return {
            "available_models": models_status_list,
            "active_model_path": active_model_path,
        }
    except Exception as e:
        logger.error(f"Error listing models with status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/models/download", tags=["Models"])
async def download_model_route(
    request: ModelActionRequest, background_tasks: BackgroundTasks
):
    """Triggers a model download in the background."""
    model_id = request.model_id
    force = request.force_download
    logger.info(f"Received request to download model: {model_id}, Force: {force}")

    available_models = get_available_models()
    if model_id not in available_models:
        logger.warning(f"Download request for unknown model_id: {model_id}")
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found in available models."
        )

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

    return {
        "message": f"Download started for model '{model_id}'. Check status endpoint for progress."
    }


@router.post("/models/set_active", tags=["Models"])
async def set_active_model_route(
    request: ModelActionRequest,
    current_llm_service: LLMServiceDep,
):
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
        logger.warning(
            f"Cannot set model '{model_id}' active, status is: {status_info['status']}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not downloaded or failed. Status: {status_info['status']}",
        )

    try:
        # Get the required model path
        model_info = AVAILABLE_MODELS[model_id]
        model_path_to_load = os.path.join(MODELS_DIR, model_info["filename"])

        # --- Reload the LLM service with the specific model path ---
        try:
            logger.info(
                f"Attempting to reload LLMService with specific model path: {model_path_to_load}..."
            )
            current_llm_service.reload_model(model_path=model_path_to_load)
            logger.info("LLMService reloaded successfully.")
            return {
                "message": f"Model '{model_id}' set as active and LLM service reloaded."
            }
        except Exception as reload_e:
            logger.error(
                f"Failed to reload LLM service for model {model_id} ({model_path_to_load}): {reload_e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reload LLM service for model {model_id}: {reload_e}",
            )

    except Exception as e:
        logger.error(f"Error setting active model {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to set active model {model_id}: {e}"
        )


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
            return {
                "model_id": model_id,
                "filename": filename,
                "config_path": config_path,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Current model configuration not found or invalid.",
            )
    except Exception as e:
        # Log the exception e
        raise HTTPException(
            status_code=500, detail=f"Error checking current model: {str(e)}"
        )


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
        raise HTTPException(
            status_code=500, detail=f"Error checking model status: {str(e)}"
        )


# --- Chat Management Routes ---


@router.get(
    "/config/llm",
    response_model=LlmConfigResponse,
    tags=["Admin & Config"],
    summary="Get current LLM configuration",
)
async def get_llm_configuration(
    current_user: CurrentUserDep,  # Optional: Secure this endpoint
):
    """Returns the current LLM configuration settings."""
    try:
        # We can read directly from the settings object
        from app.core.config import get_settings  # Local import OK here

        settings = get_settings()
        return LlmConfigResponse(
            backend=settings.LLM_BACKEND,
            model_id=settings.MODEL_ID,
            model_path=settings.MODEL_PATH,
            max_model_len=settings.MAX_MODEL_LEN,
            gpu_count=settings.GPU_COUNT,
        )
    except Exception as e:
        logger.error(f"Error retrieving LLM configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve LLM configuration"
        )


# Define the directory for saving refinement data
REFINEMENT_DATA_DIR = Path("app/data/improvement")
REFINEMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/chats/{chat_id}/messages/{message_index}/refine",
    status_code=status.HTTP_201_CREATED,
)
async def save_refined_message(
    chat_id: str,
    message_index: int,
    chat_service: ChatServiceDep,
):
    """
    Saves a specific user-assistant message pair from a chat for refinement/improvement.
    """
    logger.info(
        f"Attempting to save refinement data for chat {chat_id}, message index {message_index}"
    )
    try:
        messages = await chat_service.get_chat_history(chat_id)
        if not messages:
            logger.warning(f"Chat history not found or empty for chat_id: {chat_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat history not found or empty",
            )

        if message_index < 0 or message_index >= len(messages):
            logger.warning(
                f"Invalid message index {message_index} for chat {chat_id} with {len(messages)} messages."
            )
            # Correct indentation
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid message index. Must be between 0 and {len(messages) - 1}.",
            )
        
        # O3 Fix: Find nearest user message before the target assistant message
        # --- Corrected try/except structure ---
        user_message = None
        assistant_message = None
        try:
            assistant_message = messages[message_index]
            if assistant_message.get("role") != "assistant":
                logger.warning(
                    f"Message at index {message_index} is not an assistant message in chat {chat_id}. Role: {assistant_message.get('role')}"
                )
                # Correct indentation
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message index does not point to an assistant message.",
                )
            
            # Search backwards from the assistant message index for the nearest user message
            user_message = next(
                (
                    m
                    for m in reversed(messages[:message_index])
                    if m.get("role") == "user"
                ),
                None,  # Default to None if no preceding user message is found
            )
            
            if user_message is None:
                logger.warning(
                    f"Could not find preceding user message for index {message_index} in chat {chat_id}. Cannot save pair."
                )
                # Correct indentation
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Could not find corresponding user message for refinement.",
                )

        except IndexError:
            # Correct indentation
            logger.warning(
                f"IndexError accessing message at index {message_index} in chat {chat_id} (length {len(messages)})."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid message index."
            )
        # Note: StopIteration is handled by the user_message is None check now

        refinement_data = {
            "chat_id": chat_id,
            "user_message": user_message.get("content"),
            "assistant_response": assistant_message.get("content"),
            "timestamp": datetime.now().isoformat(),
        }

        filename = f"refinement_{chat_id}_{message_index - 1}_{message_index}.json"
        filepath = REFINEMENT_DATA_DIR / filename

        try:
            with open(filepath, "w") as f:
                json.dump(refinement_data, f, indent=2)
            logger.info(f"Successfully saved refinement data to {filepath}")
            return {
                "message": "Refinement data saved successfully",
                "filename": filename,
            }
        except IOError as e:
            logger.error(f"Failed to write refinement data to {filepath}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save refinement data",
            )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while saving refinement data for chat {chat_id}, index {message_index}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred.",
        )


@router.post(
    "/projects/{project_id}/chats",
    response_model=ChatSummaryResponse,
    tags=["Chats"],
    summary="Create a new chat within a project, checking access.",
    response_model_by_alias=True,  # Changed from False
    status_code=status.HTTP_201_CREATED,
)
async def create_new_chat_for_project(
    project_id: str,  # Path param
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    chat_data: Optional[ChatCreateRequest] = Body(None),  # Explicit Body, optional
):
    try:
        project = await asyncio.to_thread(project_manager.get_project, project_id)
        if not project:
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )
        if not await asyncio.to_thread(
            project_manager.can_access_project, project_id, current_user["id"]
        ):
            raise HTTPException(
                status_code=403, detail="Forbidden to create chat in this project."
            )

        chat_name = chat_data.name if chat_data and chat_data.name else None
        # Ensure create_chat is called with user_id now
        chat_info = await chat_service.create_chat(
            user_id=current_user["id"],  # Pass user_id
            project_id=project_id, 
            chat_name=chat_name,
        )

        if not chat_info or "chat_id" not in chat_info:
            logger.error(
                f"Chat service failed to return valid info for project {project_id}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to create chat or get valid info."
            )

        return ChatSummaryResponse(
            chat_id=chat_info["chat_id"],
            project_id=project_id,
            name=chat_info.get("name", f"Chat {chat_info['chat_id'][:8]}"),
            created_at=chat_info["created_at"],
            last_updated_at=chat_info.get("last_updated", chat_info["created_at"]),
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            f"Error creating chat for project {project_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")


@router.get(
    "/projects/{project_id}/chats",
    response_model=List[ChatSummaryResponse],
    tags=["Chats"],
    summary="List chats for a specific project, checking access.",
)
async def list_chats_for_project_route(
    project_id: str,  # Path param
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    limit: int = Query(50, ge=1, le=200),  # Query param with default
    offset: int = Query(0, ge=0),  # Query param with default
):
    try:
        project = await asyncio.to_thread(project_manager.get_project, project_id)
        if not project:
            raise HTTPException(
                status_code=404, detail=f"Project {project_id} not found"
            )
        if not await asyncio.to_thread(
            project_manager.can_access_project, project_id, current_user["id"]
        ):
            raise HTTPException(
                status_code=403, detail="Forbidden to list chats for this project."
            )

        # Ensure await is present before the async service call
        chats_data = await chat_service.list_chats_for_project(
            project_id, limit=limit, offset=offset
        )

        # Error occurs here if chats_data is a coroutine (await was missing)
        logger.info(f"Found {len(chats_data)} chats for project {project_id}")

        chat_summaries = [
            ChatSummaryResponse(
                chat_id=chat.get("chat_id"),
                project_id=chat.get("project_id"),
                name=chat.get("name", f"Chat {chat.get('chat_id','-')[:8]}"),
                created_at=chat.get("created_at"),
                last_updated_at=chat.get("last_updated", chat.get("created_at")),
            )
            for chat in chats_data
            if chat.get("chat_id")
        ]
        return chat_summaries
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            f"Error listing chats for project {project_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to list chats: {str(e)}")


@router.get(
    "/chats/{chat_id}/history",
    response_model=ChatHistoryResponse,
    tags=["Chats"],
    summary="Get chat history, checking access",
)
async def get_chat_history_route(
    chat_id: str,
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    limit: int = Query(100, ge=1, le=1000),
):
    try:
        chat_summary = await chat_service.get_chat_summary(chat_id)
        if not chat_summary or not chat_summary.get("project_id"):
            raise HTTPException(
                status_code=404, detail=f"Chat {chat_id} not found or no project ID."
            )

        project_id = chat_summary["project_id"]
        if not await asyncio.to_thread(
            project_manager.can_access_project, project_id, current_user["id"]
        ):
            raise HTTPException(
                status_code=403, detail="Forbidden to access this chat history."
            )

        history_messages = await chat_service.get_chat_history(chat_id, limit=limit)

        return ChatHistoryResponse(chat_id=chat_id, messages=history_messages)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error getting chat history for {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat history: {str(e)}"
        )


@router.post(
    "/chats/{chat_id}/messages",
    response_model=ChatMessageResponse,
    tags=["Chats"],
    summary="Add a message to a chat, checking access",
    status_code=status.HTTP_201_CREATED,
)
async def add_message_to_chat_route(
    chat_id: str,
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
    message_data: ChatMessageCreateRequest = Body(...),  # Explicit Body, required
):
    try:
        chat_summary = await chat_service.get_chat_summary(chat_id)
        if not chat_summary or not chat_summary.get("project_id"):
            raise HTTPException(
                status_code=404, detail=f"Chat {chat_id} not found or no project ID."
            )

        project_id = chat_summary["project_id"]
        if not await asyncio.to_thread(
            project_manager.can_access_project, project_id, current_user["id"]
        ):
            raise HTTPException(
                status_code=403, detail="Forbidden to add messages to this chat."
            )

        success = await chat_service.add_message_to_chat(
            chat_id, role=message_data.role, content=message_data.content
        )
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to add message to chat."
            )

        return ChatMessageResponse(
            role=message_data.role,
            content=message_data.content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error adding message to chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")


@router.delete(
    "/projects/{project_id}/chats/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Chats"],
    summary="Delete a chat session, checking access via project.",
)
async def delete_chat_route(
    project_id: str,
    chat_id: str,
    project_manager: ProjectManagerDep,
    chat_service: ChatServiceDep,
    current_user: CurrentUserDep,
):
    try:
        if not await asyncio.to_thread(
            project_manager.can_access_project, project_id, current_user["id"]
        ):
            raise HTTPException(
                status_code=403, detail="Forbidden to delete chats in this project."
            )

        chat_summary = await chat_service.get_chat_summary(chat_id)
        if not chat_summary:
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found.")
        if chat_summary.get("project_id") != project_id:
            raise HTTPException(
                status_code=403,
                detail=f"Chat {chat_id} does not belong to project {project_id}.",
            )

        deleted = await chat_service.delete_chat(chat_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} could not be deleted or was not found.",
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            f"Error deleting chat {chat_id} for project {project_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")


# --- Document Refinement Route ---
@router.post(
    "/documents/{document_id}/refine",
    response_model=ProjectDocument,  # Using ProjectDocument as DocumentResponse
    summary="Refine an existing document using AI",
    tags=["Documents"],
)
async def refine_document_route(
    document_id: str,
    refine_request: RefineRequest,
    current_user: CurrentUserDep,
    agent_runner: AgentRunnerDep,
    project_manager: ProjectManagerDep,  # To check access
):
    """
    Refines an existing document based on user prompts using the AgentRunner.

    - **document_id**: The ID of the document to refine.
    - **refine_request**: Contains the user prompt and embedding replacement flag.
    """
    # Check if user has access to the project this document belongs to
    document = project_manager.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    project_id = document.get("project_id")
    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is not associated with a project",
        )

    if not project_manager.can_access_project(project_id, current_user["id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project containing this document",
        )

    try:
        updated_document_dict = await agent_runner.refine_document(
            doc_id=document_id,
            user_prompt=refine_request.prompt,
            replace_embeddings=refine_request.replace_embeddings,
        )
        if not updated_document_dict:
            # AgentRunner might return None if the base document wasn't found by project_manager within the agent,
            # or if LLM failed. AgentRunner logs these errors.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to refine document. Check server logs for details.",
            )
        return ProjectDocument(**updated_document_dict)
    except HTTPException:  # Re-raise known HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error refining document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while refining the document: {str(e)}",
        )
