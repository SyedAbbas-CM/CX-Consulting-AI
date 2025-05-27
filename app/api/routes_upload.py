import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# from app.services import auth_service as auth # Placeholder, adjust if necessary
from app.api.dependencies import (  # Corrected import for current_user
    AuthenticatedUserDep,
)

# Assuming settings and auth are in app.core and app.services respectively
from app.core.config import get_settings
from app.services.auth_service import (
    User as AuthUser,  # Import User model for type hint
)
from app.services.ingest_worker import ingest_job
from app.utils.job_tracker import job_tracker

settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)


class UploadJobResponse(BaseModel):
    job_id: str
    message: str = "Files received and queued for processing."


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    project_id: Optional[str] = None
    progress: int = 0
    total_files: int = 0
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    details: List[str] = []
    error: Optional[str] = None


# Ensure UPLOAD_DIR exists
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload", response_model=UploadJobResponse, status_code=202)
async def upload_docs_api(
    background_tasks: BackgroundTasks,
    project_id: str = Form(...),
    files: List[UploadFile] = File(...),
    current_user: AuthUser = Depends(AuthenticatedUserDep),
):
    job_id = uuid.uuid4().hex
    upload_base_dir = Path(settings.UPLOAD_DIR)
    if not upload_base_dir.is_absolute():
        base_dir = getattr(settings, "BASE_DIR", Path.cwd())
        upload_base_dir = Path(base_dir) / settings.UPLOAD_DIR
        upload_base_dir.mkdir(parents=True, exist_ok=True)

    save_dir = upload_base_dir / job_id
    save_dir.mkdir(parents=True, exist_ok=True)

    file_details_for_task = []

    try:
        user_id = str(
            current_user.id if hasattr(current_user, "id") else current_user.get("sub")
        )
        if not user_id:
            raise ValueError(
                "User ID could not be extracted from current_user object/dict."
            )
    except Exception as e:
        logger.error(f"Could not determine user ID: {e}", exc_info=True)
        raise HTTPException(
            status_code=400, detail="Could not determine user ID from token."
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    job_tracker.start(
        job_id=job_id, user_id=user_id, total_files=len(files), project_id=project_id
    )
    job_tracker.set_processing(job_id)

    for f in files:
        if not f.filename:
            logger.warning(
                f"Skipping a file for job {job_id} because it has no filename."
            )
            job_tracker.mark_file_progress(
                job_id, "<unknown_file>", success=False, message="File has no name."
            )
            continue

        safe_filename = Path(f.filename).name
        dst = save_dir / safe_filename

        try:
            with dst.open("wb") as out:
                content = await f.read(settings.MAX_UPLOAD_SIZE_PER_FILE + 1)
                if len(content) > settings.MAX_UPLOAD_SIZE_PER_FILE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{safe_filename}' exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_PER_FILE // (1024*1024)}MB.",
                    )
                out.write(content)
                await f.seek(0)

            file_details_for_task.append(
                {
                    "filename": safe_filename,
                    "path": str(dst),
                    "content_type": f.content_type or "application/octet-stream",
                }
            )
        except HTTPException as http_exc:
            job_tracker.mark_file_progress(
                job_id, safe_filename, success=False, message=http_exc.detail
            )
            for detail in file_details_for_task:
                Path(detail["path"]).unlink(missing_ok=True)
            if save_dir.exists():
                for item in save_dir.iterdir():
                    item.unlink()
                save_dir.rmdir()
            job_tracker.finish(job_id, f"Upload failed: {http_exc.detail}")
            raise http_exc
        except Exception as e:
            logger.error(
                f"Error saving file {safe_filename} for job {job_id}: {e}",
                exc_info=True,
            )
            job_tracker.mark_file_progress(
                job_id, safe_filename, success=False, message=str(e)
            )

    if not file_details_for_task:
        job_tracker.finish(job_id, "No files could be saved for processing.")
        try:
            if save_dir.exists() and not any(save_dir.iterdir()):
                save_dir.rmdir()
        except Exception as e_clean:
            logger.error(
                f"Error cleaning up empty save directory {save_dir} for job {job_id}: {e_clean}"
            )
        raise HTTPException(
            status_code=400,
            detail="No files were successfully saved for processing. Check individual file errors.",
        )

    background_tasks.add_task(
        ingest_job,
        job_id=job_id,
        project_id=project_id,
        folder_path=save_dir,
        user_id=user_id,
        uploaded_files_metadata=file_details_for_task,
    )
    return UploadJobResponse(job_id=job_id)


@router.get("/documents/status/{job_id}", response_model=JobStatusResponse)
async def get_upload_status_api(
    job_id: str, current_user: AuthUser = Depends(AuthenticatedUserDep)
):
    logger.debug(f"Accessing job status for {job_id} by user {current_user.id}")
    job_info = job_tracker.get_status(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found.")

    user_id_from_token = str(
        current_user.id if hasattr(current_user, "id") else current_user.get("sub")
    )
    if not job_tracker.is_job_owner(job_id, user_id_from_token):
        raise HTTPException(
            status_code=403, detail="You are not authorized to view this job's status."
        )

    return JobStatusResponse(**job_info)
