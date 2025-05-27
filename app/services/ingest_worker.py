import asyncio  # Import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings  # For chunking parameters
from app.services.document_service import (  # Assuming singleton or factory access
    DocumentService,
)
from app.services.project_manager import ProjectManager  # For saving document metadata
from app.utils.extractors import extract_text_from_file
from app.utils.job_tracker import job_tracker  # Import the global instance


# A way to get DocumentService and ProjectManager instances.
# This might be via a global factory, app state, or passed in if ingest_job becomes a class method.
# For now, let's assume they can be instantiated or fetched as needed.
# This is a placeholder and needs to align with your app's service management.
def get_document_service_instance() -> DocumentService:
    from app.main import app  # Example: fetching from app.state

    if not hasattr(app.state, "document_service") or not app.state.document_service:
        raise RuntimeError("DocumentService not initialized in app.state.")
    return app.state.document_service


def get_project_manager_instance() -> ProjectManager:
    from app.main import app  # Example: fetching from app.state

    if not hasattr(app.state, "project_manager") or not app.state.project_manager:
        raise RuntimeError("ProjectManager not initialized in app.state.")
    return app.state.project_manager


logger = logging.getLogger(__name__)
settings = get_settings()


async def ingest_job(
    job_id: str,
    project_id: str,  # Can be None if is_global_upload is True
    folder_path: Path,
    user_id: str,
    uploaded_files_metadata: List[Dict[str, str]],
    is_global_upload: bool = False,  # <--- Added new parameter with default
) -> None:
    logger.info(
        f"Starting ingest_job {job_id} for project '{project_id if project_id else '_global_'}' (is_global: {is_global_upload}), user {user_id}. Folder: {folder_path}"
    )
    # Update job tracker start to reflect potential global upload
    job_tracker.start(
        job_id=job_id,
        user_id=user_id,
        total_files=len(uploaded_files_metadata),
        project_id=project_id if not is_global_upload else "_global_",
    )
    job_tracker.set_processing(job_id)  # Mark job as actively processing

    doc_service: Optional[DocumentService] = None
    proj_manager: Optional[ProjectManager] = None

    try:
        # These are synchronous, so no await needed here
        doc_service = get_document_service_instance()
        proj_manager = get_project_manager_instance()
    except RuntimeError as e:
        logger.error(
            f"Failed to get service instances for job {job_id}: {e}", exc_info=True
        )
        job_tracker.finish(
            job_id, f"Internal server error: Could not initialize services. {e}"
        )
        # Consider cleanup of folder_path contents if services can't be obtained
        return
    except Exception as e_services:
        logger.error(
            f"Unexpected error getting service instances for job {job_id}: {e_services}",
            exc_info=True,
        )
        job_tracker.finish(job_id, f"Unexpected internal server error: {e_services}")
        return

    processed_file_count = 0
    for (
        file_meta
    ) in uploaded_files_metadata:  # Iterate using the metadata from upload_docs_api
        file_path = Path(file_meta["path"])
        original_filename = file_meta["filename"]
        content_type = file_meta.get("content_type", "application/octet-stream")

        logger.info(
            f"Processing file: {original_filename} (path: {file_path}) for job {job_id}"
        )

        try:
            # extract_text_from_file is synchronous
            text_content, extracted_meta = await asyncio.to_thread(
                extract_text_from_file, file_path
            )

            if (
                not text_content
                and "extraction_error" not in extracted_meta
                and "extraction_warning" not in extracted_meta
            ):
                logger.warning(
                    f"File {original_filename} yielded no text content. Skipping indexing."
                )
                job_tracker.mark_file_progress(
                    job_id,
                    original_filename,
                    success=False,
                    message="File yielded no text or is empty.",
                )
                continue  # Skip to next file
            elif "extraction_error" in extracted_meta:
                logger.error(
                    f"Failed to extract text from {original_filename}: {extracted_meta['extraction_error']}"
                )
                job_tracker.mark_file_progress(
                    job_id,
                    original_filename,
                    success=False,
                    message=f"Extraction failed: {extracted_meta['extraction_error']}",
                )
                continue  # Skip to next file

            # Create a new document ID for this uploaded file
            document_id = str(uuid.uuid4())

            # Prepare document metadata for ProjectManager and DocumentService
            # The document_type from extracted_meta might be more specific (e.g., "pdf", "docx")
            # We need a more general "user_upload" or similar for the high-level doc type in ProjectManager.
            doc_pm_type = "user_upload"  # General type for ProjectManager
            doc_ds_type = extracted_meta.get(
                "document_type",
                Path(original_filename).suffix.lstrip(".").lower() or "unknown",
            )  # More specific for DS

            doc_metadata_for_pm = {
                "source_filename": original_filename,
                "job_id": job_id,
                "uploader_user_id": user_id,
                "content_type": content_type,
                "file_size_bytes": (
                    file_path.stat().st_size if file_path.exists() else 0
                ),
                **extracted_meta,  # Add all metadata from the extractor
            }

            # 1. Save document metadata using ProjectManager
            # This creates the initial record of the document in the project.
            # `create_document` in ProjectManager should not itself trigger indexing.
            await asyncio.to_thread(
                proj_manager.create_document,
                project_id=project_id,
                title=original_filename,  # Use filename as initial title
                content="",  # Content will be indexed by DocumentService, not stored directly here in full
                document_type=doc_pm_type,  # e.g., "user_upload"
                metadata=doc_metadata_for_pm,
                user_id=user_id,
                # conversation_id can be None for direct uploads
            )
            logger.info(
                f"Created document record {document_id} for {original_filename} in project {project_id}"
            )

            # 2. Index content using DocumentService
            # This will chunk and add to vector store. `index_document_content` uses its own collection policy.
            doc_metadata_for_indexing = {
                "project_id": project_id,
                "document_type": doc_ds_type,  # e.g., "pdf", "docx" or from extractor
                "source": f"project_doc:{document_id}",  # Link to the PM document ID
                "title": original_filename,
                "original_filename": original_filename,  # Redundant but can be useful
                "created_at": time.time(),  # DocumentService might manage its own processed_at
                "user_id": user_id,  # For ownership/filtering in vector store if needed
                # Pass through any other relevant metadata from extraction that DS might use
                **extracted_meta,
            }

            # We are indexing the content associated with the *new* document_id managed by ProjectManager
            success_indexing = await doc_service.index_document_content(
                doc_id=document_id,  # Use the ProjectManager's document_id as the reference
                content=text_content,
                doc_metadata=doc_metadata_for_indexing,
            )

            if success_indexing:
                logger.info(
                    f"Successfully indexed content for {original_filename} (doc_id: {document_id}) in job {job_id}"
                )
                job_tracker.mark_file_progress(job_id, original_filename, success=True)
            else:
                logger.error(
                    f"Failed to index content for {original_filename} (doc_id: {document_id}) in job {job_id}"
                )
                job_tracker.mark_file_progress(
                    job_id,
                    original_filename,
                    success=False,
                    message="Failed during content indexing phase.",
                )
                # Consider if the document record in ProjectManager should be marked as failed or removed

        except Exception as e_file_processing:
            logger.error(
                f"Unhandled error processing file {original_filename} in job {job_id}: {e_file_processing}",
                exc_info=True,
            )
            job_tracker.mark_file_progress(
                job_id,
                original_filename,
                success=False,
                message=f"Unexpected error: {e_file_processing}",
            )
        finally:
            processed_file_count += 1
            # Optionally, delete the local file from save_dir after processing to save space, if not needed later
            # try:
            #     file_path.unlink(missing_ok=True)
            # except Exception as e_unlink:
            #     logger.warning(f"Could not delete temp file {file_path} for job {job_id}: {e_unlink}")

    # After all files are processed
    job_tracker.finish(job_id)
    logger.info(
        f"Ingest_job {job_id} finished. Processed {processed_file_count} files."
    )

    # Optional: Clean up the job_id folder in UPLOAD_DIR if all files were successfully processed and deleted individually
    # Or if it's meant to be transient storage. This needs careful consideration.
    # For now, we leave the folder as it might contain logs or failed files for inspection.
    # If you want to clean up:
    # try:
    #     if folder_path.exists():
    #         import shutil
    #         shutil.rmtree(folder_path)
    #         logger.info(f"Cleaned up job folder: {folder_path} for job {job_id}")
    # except Exception as e_cleanup:
    #     logger.error(f"Error cleaning up job folder {folder_path} for job {job_id}: {e_cleanup}")
