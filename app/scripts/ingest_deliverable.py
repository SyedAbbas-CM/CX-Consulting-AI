import argparse
import asyncio
import json
import logging
import os
import re  # For slugifying
import sys
from datetime import datetime  # For doc_id timestamp

# Add project root to sys.path to allow importing app modules
from pathlib import Path
from typing import Any, Dict, List, Optional  # Add Optional here

from dotenv import load_dotenv

# Adjust the Python path to include the root directory of the application
# This allows us to import modules from the app package (e.g., app.services)
# Get the absolute path of the current script (ingest_deliverable.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the project root (assuming ingest_deliverable.py is in app/scripts/)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.core.config import get_settings
from app.services.document_service import DELIVERABLE_COLLECTION_PREFIX, DocumentService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Source directory for deliverable guide documents - base directory containing type subfolders
SOURCE_DOCS_DIR = project_root / "data" / "deliverable_docs"
ALLOWED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"]
# -------------------


def slugify(value: str) -> str:
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(value)
    value = value.lower()
    value = re.sub(
        r"[^\\w\\s-]", "", value
    )  # Remove non-alphanumeric (excluding hyphens and spaces)
    value = re.sub(r"[-\\s]+", "_", value).strip(
        "_"
    )  # Convert spaces/hyphens to single underscore
    return value


async def ingest_deliverable_guides():
    """Scans subdirectories in SOURCE_DOCS_DIR, treating each as a deliverable type,
    and ingests supported documents into dynamically named collections using DocumentService.add_document().
    """
    logger.info(
        f"Starting ingestion of deliverable guides from base directory: {SOURCE_DOCS_DIR}"
    )

    if not SOURCE_DOCS_DIR.exists() or not SOURCE_DOCS_DIR.is_dir():
        logger.error(
            f"Base source directory does not exist or is not a directory: {SOURCE_DOCS_DIR}"
        )
        return

    settings = get_settings()
    doc_service: Optional[DocumentService] = None

    try:
        # Initialize DocumentService once
        # Note: DocumentService's async_init should be called if it's a separate step.
        # If DocumentService.__init__ can be async or handles its own async setup, this is fine.
        # For now, assuming __init__ is synchronous and internal async setup happens on first use or via a dedicated async_init method.
        # Based on current DocumentService, we'll need to call its async setup.
        doc_service = DocumentService(
            vectorstore_dir=str(settings.VECTOR_DB_PATH),
            embedding_model_name_or_path=settings.EMBEDDING_MODEL_ID,
            default_collection_name=settings.DEFAULT_CHROMA_COLLECTION,  # A default, though we override
            bm25_dir=str(
                settings.VECTOR_DB_PATH / "bm25_data"
            ),  # Explicitly pass bm25_dir
        )
        # Explicitly initialize embeddings and vector store components if not done in __init__
        # or if add_document relies on these being ready.
        # DocumentService.add_document internally calls _init_embedding_model and _init_vector_store for the target collection.
        logger.info("DocumentService initialized for ingestion.")

    except Exception as e:
        logger.error(f"Failed to initialize DocumentService: {e}", exc_info=True)
        return

    if not doc_service:
        logger.error("DocumentService is not available after initialization attempt.")
        return

    total_processed_files = 0
    total_failed_files = 0

    for type_dir in SOURCE_DOCS_DIR.iterdir():
        if not type_dir.is_dir():
            logger.debug(f"Skipping non-directory item: {type_dir.name}")
            continue

        deliverable_type_original = type_dir.name
        deliverable_type_slug = slugify(deliverable_type_original)
        target_collection_name = (
            f"{DELIVERABLE_COLLECTION_PREFIX}{deliverable_type_slug}"
        )

        logger.info(
            f"Processing deliverable type: '{deliverable_type_original}' (slug: '{deliverable_type_slug}')"
        )
        logger.info(f"Target collection for this type: '{target_collection_name}'")

        processed_in_type = 0
        failed_in_type = 0

        logger.info(
            f"Scanning for files in {type_dir} with extensions: {ALLOWED_EXTENSIONS}"
        )
        for filepath in type_dir.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in ALLOWED_EXTENSIONS:
                logger.info(f"Found candidate file: {filepath.name} in {type_dir.name}")

                doc_file_type_ext = filepath.suffix.lower().lstrip(".")
                relative_path_str = str(filepath.relative_to(project_root))

                metadata_for_doc = {
                    "filename": filepath.name,
                    "deliverable_type": deliverable_type_original,
                    # "source" is automatically added by add_document from document_url if it's a local path
                    # Let's ensure it's the relative path we want.
                    # add_document also creates its own doc_id.
                }

                try:
                    logger.info(
                        f"Processing file via DocumentService.add_document: {filepath}"
                    )
                    # DocumentService.add_document handles text extraction, chunking, embedding, and storage.
                    # It needs the full path as `document_url`.
                    # `project_id` is None for these general deliverable guides.
                    # `is_global` is False as these are deliverable-specific, not globally applicable in the same way as global_kb.
                    result = await doc_service.add_document(
                        document_url=str(
                            filepath.resolve()
                        ),  # Use resolved absolute path
                        document_type=doc_file_type_ext,
                        collection_name=target_collection_name,
                        metadata=metadata_for_doc,
                        is_global=False,  # These are not part of the global_kb
                        project_id=None,  # Not tied to a specific project instance
                    )

                    if result and result.get("status") == "success":
                        logger.info(
                            f"Successfully processed and added '{filepath.name}' (ID: {result.get('document_id')}) to collection '{target_collection_name}'. Chunks: {result.get('chunks_created')}"
                        )
                        processed_in_type += 1
                    else:
                        logger.error(
                            f"Failed to process file {filepath.name} for type {deliverable_type_original} using add_document. Result: {result}",
                        )
                        failed_in_type += 1

                except Exception as e:
                    logger.error(
                        f"Exception processing file {filepath.name} for type {deliverable_type_original}: {e}",
                        exc_info=True,
                    )
                    failed_in_type += 1
            elif filepath.is_file():
                logger.debug(
                    f"Skipping file with unsupported extension: {filepath.name} in {type_dir.name}"
                )

        logger.info(f"--- Summary for Type: {deliverable_type_original} ---")
        logger.info(f"  Processed files: {processed_in_type}")
        logger.info(f"  Failed files: {failed_in_type}")
        total_processed_files += processed_in_type
        total_failed_files += failed_in_type

    logger.info(f"--- Overall Ingestion Summary ---")
    logger.info(f"Total processed files across all types: {total_processed_files}")
    logger.info(f"Total failed files across all types: {total_failed_files}")
    logger.info(f"Deliverable guides ingestion finished.")


if __name__ == "__main__":
    asyncio.run(ingest_deliverable_guides())
