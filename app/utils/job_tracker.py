import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobTracker:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = (
            {}
        )  # job_id -> {status, details, error, progress, user_id, etc.}

    def start(self, job_id: str, user_id: str, total_files: int, project_id: str):
        self.jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_id,
            "project_id": project_id,  # Store project_id for the job
            "status": "queued",
            "details": [f"Job {job_id} created. {total_files} file(s) to process."],
            "error": None,
            "progress": 0,
            "total_files": total_files,
            "files_processed": 0,
            "files_succeeded": 0,
            "files_failed": 0,
        }
        logger.info(
            f"Job {job_id} queued for user {user_id}, project {project_id}. Total files: {total_files}"
        )

    def set_processing(self, job_id: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "processing"
            logger.info(f"Job {job_id} changed status to processing.")
        else:
            logger.error(f"Tried to set status for unknown job_id: {job_id}")

    def mark_file_progress(
        self, job_id: str, filename: str, success: bool, message: Optional[str] = None
    ):
        if job_id not in self.jobs:
            logger.error(f"Tried to mark progress for unknown job_id: {job_id}")
            return

        job = self.jobs[job_id]
        job["files_processed"] += 1
        if success:
            job["files_succeeded"] += 1
            job["details"].append(f"Successfully processed: {filename}")
        else:
            job["files_failed"] += 1
            detail_msg = f"Failed to process: {filename}"
            if message:
                detail_msg += f" - Error: {message}"
            job["details"].append(detail_msg)
            logger.warning(
                f"File processing failed for job {job_id}, file {filename}: {message}"
            )

        if job["total_files"] > 0:
            job["progress"] = int((job["files_processed"] / job["total_files"]) * 100)
        elif (
            job["files_processed"] > 0
        ):  # If total_files was 0 but we processed some (should not happen)
            job["progress"] = 100
        else:
            job["progress"] = 0

    def finish(self, job_id: str, error_message: Optional[str] = None):
        if job_id not in self.jobs:
            logger.error(f"Tried to finish unknown job_id: {job_id}")
            return

        job = self.jobs[job_id]
        if error_message:
            job["status"] = "failed"
            job["error"] = error_message
            job["details"].append(f"Job failed: {error_message}")
            logger.error(f"Job {job_id} finished with error: {error_message}")
        elif job["files_failed"] > 0:
            job["status"] = "completed_with_errors"
            job["error"] = (
                f"{job['files_failed']} out of {job['total_files']} files failed to process."
            )
            job["details"].append(f"Job completed with {job['files_failed']} errors.")
            logger.warning(f"Job {job_id} completed with {job['files_failed']} errors.")
        else:
            job["status"] = "completed"
            job["details"].append("Job completed successfully.")
            logger.info(f"Job {job_id} finished successfully.")
        job["progress"] = 100

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def is_job_owner(self, job_id: str, user_id: str) -> bool:
        job_info = self.jobs.get(job_id)
        return job_info is not None and job_info.get("user_id") == user_id


# Global instance (or manage through FastAPI app state / dependency injection for better practice)
job_tracker = JobTracker()
